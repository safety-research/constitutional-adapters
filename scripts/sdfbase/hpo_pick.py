#!/usr/bin/env python
"""hpo_pick — record the operator's pick of the LoRA architecture in two
append-only phases, and hand the fit step its selection.

    python scripts/sdfbase/hpo_pick.py --search-dir <hpo root> --rank R --set S \
        [--allowed-pairs allowed_pairs.json] [--models-table reproduce/models.yaml] \
        [--model-key qwen35_4b_demo] [--fit-config configs/sdfbase_<key>_fits.yaml] \
        [--table-url URL] [--ruled-utc ISO] [--execute]
    python scripts/sdfbase/hpo_pick.py --search-dir <hpo root> --bind \
        --object-ids <control_lora_id>,<constitution_lora_id>,<difference_lora_id> [--execute]

Phase A (the pick): REFUSES a pair outside the allowed pairs P (excluded by
the probe, or never probed); composes the `registry ruling add` argv
(model-scoped, action operating_point, provenance user_direct, id
`r2-operating_point-<model>-lora-architecture-selection`) and EXECUTES it
only under --execute (otherwise the argv is printed, so a dry run shows the
exact record); updates `models.yaml` `lora_architecture.{rank, modules,
per_expert_rank, selected_by: user_pick, selection_ruling_id}` by a tolerant
line-level round-trip that leaves every other key and comment byte-
identical; writes `hpo_pick.json` beside the search — the input
`make_sdfbase_configs.py --fold-hpo` consumes:

    {model_key: {"lora_r4_ce": {lr, warmup_frac, max_grad_norm, lora_rank,
        lora_modules, per_expert_rank, init_scheme, fit_search_id,
        architecture_selection_ruling_id, fit_lr_selected_by: "manual"}}}

and closes the fit_search with the pick as winner_params (emit_hpo.
close_search, lazily; absent -> a printed notice). The pick's lr comes from
the pair-only refit (`r1+2_fit_pair.json`: the refit's argmin, or the
promoted measured trial) when it exists, else from the design fit's
prediction at that pair, else from the pair's best measured trial — and the
source is recorded.

Phase B (--bind): once the three LoRA object shas exist, composes the
object-scoped ruling that binds phase A to them (idempotent: the same id).

No default is applied for either: --rank and --set are required for phase A;
the stage script refuses without them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ONE tune_arm instance per process (the tests register theirs as `_tune_arm`
# — hpo_next_round's convention); the guard rides on it.
tune_arm = (sys.modules.get("_tune_arm") or sys.modules.get("tune_arm")
            or _load_by_path("tune_arm", SCRIPTS / "tune_arm.py"))
guard_mod = tune_arm.hpo_selection_guard

PHASE_A_ID = "r2-operating_point-{model}-lora-architecture-selection"
PHASE_B_ID = "r2-operating_point-{model}-lora-architecture-objects"
ARM = "lora_r4_ce"
#: The CONFIRM-PICK request: an off-winner pick whose pair has
#: no measured clip evidence at its learning rate is recorded (the ruling) but NOT promoted — this file asks
#: `03_hpo.sh --phase confirm-pick` for ONE trial there (`tune_arm.py emit-trials --confirm-pick-from`);
#: `--phase pick` re-runs the gate once its summary lands. The pointer 03 reads is CONFIRM_PICK_REQUEST_NAME
#: (the CURRENT request); every request also lives at its own IMMUTABLE per-index file (the ledger,
#: confirm_pick_request_name(index)) and reserves, at request time, the identity its trial will run under:
#: `index` (one past every reserved / landed confirm trial), `design` = r8_tNN_trials.json (the per-request
#: design path, the same basename on the grading host and the cluster, never a shared r8_trials.json),
#: `trial_id` = r8_tNN, `job_suffix` = r8-tNN (03 prefixes hpo-<model>-<class>-).
#: A queued job's design input is therefore immutable, and a later, different pick can never share its index,
#: its design file or its job name; 03 refuses to submit a replacement while an earlier confirm-pick job is active.
CONFIRM_PICK_REQUEST_NAME = "hpo_pick_confirm_request.json"
CONFIRM_PICK_LEDGER_GLOB = "hpo_pick_confirm_request_t*.json"


def confirm_pick_request_name(index: int) -> str:
    """The immutable per-request file (the ledger entry) for confirm-pick index ``index``."""
    return f"hpo_pick_confirm_request_t{int(index):02d}.json"


def confirm_pick_identity(index: int) -> dict:
    """The identity a confirm-pick request reserves for its trial: the trial id, the per-request design path
    (basename) and the job-name suffix — all derived from the index and nothing else, so 03_hpo.sh and
    hpo_trial.sbatch (which derives its trial dir from the design row's index) agree by construction."""
    r = tune_arm.CONFIRM_PICK_ROUND
    return {"index": int(index), "trial_id": f"r{r}_t{int(index):02d}",
            "design": f"r{r}_t{int(index):02d}_trials.json", "job_suffix": f"r{r}-t{int(index):02d}"}


def confirm_pick_ledger(search_dir: Path) -> dict[int, dict]:
    """Every reserved confirm-pick request under ``search_dir`` by index (the immutable per-index files); an
    unparseable entry is refused by name — the ledger is what makes an index unique, so it is never guessed past."""
    out: dict[int, dict] = {}
    for f in sorted(Path(search_dir).glob(CONFIRM_PICK_LEDGER_GLOB)):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            raise SystemExit(f"[hpo-pick] the confirm-pick ledger entry {f} does not parse ({e}) — refusing to reserve "
                             "an index past a torn entry; repair or remove it")
        idx = d.get("index") if isinstance(d, dict) else None
        if not isinstance(idx, int) or f.name != confirm_pick_request_name(idx):
            raise SystemExit(f"[hpo-pick] the confirm-pick ledger entry {f} names index {idx!r}, not the index in its "
                             "file name — refusing (the ledger was edited by hand)")
        out[idx] = d
    return out


def next_confirm_pick_index(search_dir: Path, pending: dict | None = None) -> int:
    """One past every index already RESERVED (the ledger), LANDED (the mirror's r8_t* dirs, a pre-ledger world) or
    named by the current pointer — never a count of trial directories (a queued, unfetched trial has none, which is
    exactly how two picks collided on index 0)."""
    search_dir = Path(search_dir)
    used = set(confirm_pick_ledger(search_dir))
    for d in search_dir.glob(f"r{tune_arm.CONFIRM_PICK_ROUND}_t*"):
        tail = d.name.rsplit("_t", 1)[-1]
        if d.is_dir() and tail.isdigit():
            used.add(int(tail))
    if pending and isinstance(pending.get("index"), int):
        used.add(pending["index"])
    return (max(used) + 1) if used else 0


def utc_now() -> str:
    """The registry's timestamp spelling (tune_arm._utc_now -> records.utc_now)."""
    return tune_arm._utc_now()


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text())


HOLD_NAME = "hpo_pick.HOLD.json"      # A completed pick whose basis changed is HELD — every consumer refuses by name while this file exists


def write_hold(search_dir: Path, reason: str) -> Path:
    """The pick phase found that a fit no longer stands over the current trial evidence AFTER a promotion — the promotion is
    HELD (never silently left in force, never auto-revoked: there is no un-fold). The hold names its reason and the pick it holds."""
    import datetime as _dt
    p = Path(search_dir) / HOLD_NAME
    prior = None
    pp = Path(search_dir) / "hpo_pick.json"
    if pp.is_file():
        try:
            prior = json.loads(pp.read_text())
        except Exception:                                   # pragma: no cover
            prior = {"unreadable": str(pp)}
    doc = {"written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reason": reason,
           "held_pick": {k: (prior or {}).get(k) for k in ("rank", "set", "ruling_id", "winner_trial_id", "search_id", "ruled_utc")},
           "clears_when": "a fresh `03_hpo.sh --phase pick --rank R --set S` re-derives a promotion from the current evidence"}
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return p


def read_hold(search_dir: Path) -> dict | None:
    p = Path(search_dir) / HOLD_NAME
    return json.loads(p.read_text()) if p.is_file() else None


def annotate_hold_cleared(search_dir: Path, pick: dict) -> dict:
    """First half of the hold clearance: a fresh promotion from the current evidence records the hold it clears in its own provenance
    (`hold_cleared`: when, and the hold's content) — the hold FILE stays in place until :func:`finalize_hold_clear`, because the interlock
    must outlive every promotion write; a failure between the two leaves the hold standing and hpo_pick.json saying what it would have
    cleared."""
    import datetime as _dt
    hold = read_hold(search_dir)
    if hold is None:
        return pick
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {**pick, "hold_cleared": {"cleared_utc": utc, "previous_hold": hold, "pending": True}}


def finalize_hold_clear(search_dir: Path, *, execute: bool) -> None:
    """Second half of the hold clearance: AFTER every promotion write and the registry close succeeded, move the hold file aside (never deleted) and
    mark the recorded clearance final in hpo_pick.json. The shell's fold step runs after this; a fold failure RESTORES the hold (03)."""
    p = Path(search_dir) / HOLD_NAME
    if not p.is_file():
        return
    if not execute:
        print("DRY-RUN: this promotion would clear the HOLD (recorded as hold_cleared)")
        return
    utc = _dt_utc()
    dst = Path(search_dir) / f"hpo_pick.HOLD.cleared.{utc}.json"
    p.rename(dst)
    pp = Path(search_dir) / "hpo_pick.json"
    try:
        doc = json.loads(pp.read_text())
        if isinstance(doc.get("hold_cleared"), dict):
            doc["hold_cleared"].pop("pending", None); doc["hold_cleared"]["finalized_utc"] = utc
            pp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    except Exception:                                       # pragma: no cover — the hold is cleared; the annotation is provenance
        pass
    print(f"[hpo-pick] the HOLD is cleared by this promotion (recorded as hold_cleared; the hold moved to {dst.name})")


def _dt_utc() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def verify_promotion(search_dir: Path, allowed_path: str | None, source_record: str | None) -> list[str]:
    """A COMPLETED promotion is re-evaluated against the CURRENT evidence — every fit bound (load_search), the
    recorded (pair, lr) still what the fits select, and the clip gate at the recorded coordinates still promotable over every round's
    trials (a confirming trial's backfilled sidecar at those coordinates would otherwise leave the fold in force). Returns the reasons the
    promotion no longer stands (empty = stands)."""
    sd = Path(search_dir)
    pp = sd / "hpo_pick.json"
    if not pp.is_file():
        return [f"no promotion recorded ({pp} absent)"]
    pick = json.loads(pp.read_text())
    if not pick.get("promoted"):
        return ["hpo_pick.json records no promotion (promoted is false)"]
    key = tune_arm.pair_key(pick["rank"], pick["set"])
    search = load_search(sd, allowed_path, source_record)            # refuses by name when a fit does not stand over the evidence
    rec = pick.get("lr_source") or {}
    rec_lr = rec.get("lr")
    reasons: list[str] = []
    cur = lr_for_pair(search, key)
    if rec_lr is None:
        reasons.append("the promotion records no learning rate (lr_source.lr)")
    else:
        if not math.isclose(float(cur["lr"]), float(rec_lr), rel_tol=1e-9):
            reasons.append(f"the fits now select lr {cur['lr']:.4g} ({cur['source']}) at rank {key[0]} / set {key[1]}, the promotion shipped {float(rec_lr):.4g} ({rec.get('source')})")
        gate = clip_gate_for_pick(search, key, {"lr": float(rec_lr), "source": "the recorded promotion"})["clip_evidence"]
        if gate["verdict"] != "promotable":
            reasons.append(f"the clip gate at the promoted coordinates (rank {key[0]} / set {key[1]}, lr {float(rec_lr):.4g}) is now {gate['verdict']}: {gate['reason']}")
    # the trial the promotion RESTS on (the recorded confirmation, else the winner trial) is re-read from its CURRENT
    # summary — a retried confirming trial that now fails the floor, lost its status or ran under other fixed axes invalidates the promotion
    # even though every fit (rounds 1 / 1–2) still binds and the clip gate still passes
    conf_tid = (pick.get("confirmation") or {}).get("trial_id") or pick.get("winner_trial_id")
    if conf_tid:
        y_key = search["design"].get("y_key") or "best_heldout_loss"
        row = next((r for r in addendum_rows_at(sd, key, y_key) if r["trial_id"] == conf_tid), None)
        if row is None:
            sp = sd / conf_tid / "summary.json"
            try:
                smry = json.loads(sp.read_text()); t = smry.get("trial") or {}
                row = {"trial_id": conf_tid, "lr": t.get("lr"), "max_grad_norm": t.get("max_grad_norm"), "warmup_frac": t.get("warmup_frac"),
                       "init_scheme": t.get("init_scheme") or smry.get("init_scheme"), "status": smry.get("status"), "floor_ok": smry.get("floor_ok")}
            except (OSError, ValueError) as e:
                reasons.append(f"the trial the promotion rests on ({conf_tid}) has no readable summary on the mirror ({e})"); row = None
        if row is not None:
            wp = pick.get("winner_params") or {}
            cand = {"lr": rec_lr, "max_grad_norm": wp.get("max_grad_norm", rec.get("max_grad_norm")), "warmup_frac": wp.get("warmup_frac", rec.get("warmup_frac"))}
            why = confirmation_identity_mismatch(row, cand, pick.get("init_scheme") or search["design"].get("init_scheme"))
            if row.get("floor_ok") is not True:
                why = (why + "; " if why else "") + f"floor_ok {row.get('floor_ok')!r} (an affirmative floor pass is required of the confirming trial)"
            if row.get("lr") is None or rec_lr is None or not math.isclose(float(row["lr"]), float(rec_lr), rel_tol=1e-9):
                why = (why + "; " if why else "") + f"lr {row.get('lr')!r} vs the promoted {rec_lr!r}"
            if why:
                reasons.append(f"the trial the promotion rests on ({conf_tid}) no longer confirms it as it stands: {why}")
    # the menu the pick was chosen from is part of the promotion's identity — a widened or replaced menu holds the pick so the
    # operator chooses again from the new table; a promotion that records no menu identity is held too
    menu_rec = pick.get("menu") or {}
    cur_menu = search["allowed"] or {}
    if not menu_rec.get("sha256"):
        reasons.append("the promotion records no menu identity (menu.sha256) — the pick cannot be verified against the current menu; pick again from the current table")
    elif cur_menu.get("sha256") != menu_rec["sha256"]:
        reasons.append(f"the menu changed since the pick (menu sha {str(menu_rec['sha256'])[:16]} -> {str(cur_menu.get('sha256'))[:16]}, {menu_rec.get('n_pairs')} -> {len(cur_menu.get('pairs') or [])} pairs) — pick again from the current table")
    return reasons


def load_search(search_dir: Path, allowed_path: str | None, source_record: str | None = None) -> dict:
    design = search_dir / "r1_fit.json"
    if not design.is_file():
        raise SystemExit(f"[hpo-pick] no design fit at {design} — run "
                         "`tune_arm.py analyze --design v2` first")
    art = _load_json(design)
    refit_p = search_dir / "r1+2_fit_pair.json"
    refit = _load_json(refit_p) if refit_p.is_file() else None
    # a pick reads the fit's conclusions — they must stand over the CURRENT trial evidence (rows, objective values, clip
    # evidence from each trial's metrics.jsonl), exactly as the table requires; a fit that predates a retried trial or a metrics backfill is refused
    hpo_table = sys.modules.get("hpo_table") or _load_by_path("hpo_table", Path(__file__).resolve().parent / "hpo_table.py")
    summaries = hpo_table.load_summaries(search_dir)
    menu_p = search_dir / hpo_table.MENU_FIT_NAME
    menu_fit = _load_json(menu_p) if menu_p.is_file() else None
    for name, fit in (("r1_fit.json", art), ("r1+2_fit_pair.json", refit), (hpo_table.MENU_FIT_NAME, menu_fit)):
        why = hpo_table._fit_rows_binding(fit, summaries) if fit is not None else None
        if why:
            raise SystemExit(f"[hpo-pick] {name} {why} — refusing to pick from a fit that does not stand over the current evidence (re-run --phase analyze)")
    if menu_fit is not None:                                   # the widened-menu refit is bound here exactly as the table binds it
        allowed_doc = tune_arm.load_allowed_pairs(allowed_path) if allowed_path else None
        why = hpo_table._menu_fit_binding(menu_fit, art, allowed_doc, source_record)
        if why:
            raise SystemExit(f"[hpo-pick] {hpo_table.MENU_FIT_NAME} {why} — refusing to pick beside a widened-menu refit that is not bound to this search (re-derive it: --phase addendum)")
    trials_p = search_dir / "r1_trials.json"
    trials = _load_json(trials_p) if trials_p.is_file() else {}
    if allowed_path:
        allowed = tune_arm.load_allowed_pairs(allowed_path)
    else:
        ap = search_dir / "allowed_pairs.json"
        if ap.is_file():
            allowed = tune_arm.load_allowed_pairs(ap)
        elif trials.get("allowed_pairs_detail"):
            # the design payload carries the probe's pairs VERBATIM (rank, set,
            # lora_modules, per_expert_rank, trainable_params) and its source
            allowed = {"pairs": [dict(pr) for pr in trials["allowed_pairs_detail"]],
                       "excluded": [{"rank": e[0], "set": e[1], "reason": e[2]}
                                    for e in trials.get("excluded_pairs") or []],
                       "path": str(trials_p),
                       "sha256": (trials.get("allowed_pairs_source") or {}).get("sha256"),
                       "model_id": (trials.get("allowed_pairs_source") or {}).get("model_id")}
        elif art.get("allowed_pairs"):
            allowed = {"pairs": [{"rank": a, "set": b,
                                  "lora_modules": tune_arm.modules_of_set(b),
                                  "per_expert_rank": None}
                                 for a, b in art["allowed_pairs"]],
                       "excluded": [{"rank": e[0], "set": e[1], "reason": e[2]}
                                    for e in art.get("excluded_pairs") or []],
                       "path": str(design), "sha256": None, "model_id": None}
        else:
            raise SystemExit("[hpo-pick] no allowed-pair menu: pass --allowed-pairs "
                             "or place allowed_pairs.json beside the search")
    search_id = art.get("search_id") or trials.get("search_id")
    return {"design": art, "refit": refit, "menu_fit": menu_fit, "allowed": allowed, "trials": trials,
            "search_id": search_id, "dir": str(search_dir)}


def pair_rows(art: dict, key) -> list[dict]:
    return [r for r in art.get("rows") or [] if tune_arm._pair_of(r) == key]


def addendum_rows_at(search_dir: Path, key, y_key: str) -> list[dict]:
    """Every terminal ADDENDUM trial at the pair (round 4, trial_role addendum: the rows the table lists as evidence beside the round-1 search —
    a pair the design never drew, run at the widened-menu refit's prediction), as rows carrying the trial's own coordinates and fixed axes
    (lr, warmup_frac, max_grad_norm, init_scheme), its status / floor verdict, its objective and the clip-activation rate measured from its
    own metrics.jsonl. Such a row, when it matches the pick's coordinates, IS the pick's confirmation."""
    out: list[dict] = []
    for p in sorted(Path(search_dir).glob("r*_t*/summary.json")):
        try:
            smry = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(smry, dict) or smry.get("status") not in tune_arm._TERMINAL_STATUSES:
            continue
        t = smry.get("trial") or {}
        if (smry.get("trial_role") or t.get("trial_role")) != "addendum" or t.get("lr") is None or tune_arm._pair_of(t) != tuple(key):
            continue
        rate, n_steps = tune_arm._trial_clip_activation(p.parent, smry)
        out.append({"trial_id": p.parent.name, "round": smry.get("round"), "index": t.get("index"), "lr": float(t["lr"]),
                    "warmup_frac": t.get("warmup_frac"), "max_grad_norm": t.get("max_grad_norm"),
                    "init_scheme": t.get("init_scheme") or smry.get("init_scheme"), "status": smry.get("status"), "floor_ok": smry.get("floor_ok"),
                    y_key: smry.get(y_key), "clip_activation_rate": rate, "clip_n_steps": n_steps,
                    "lora_rank": t.get("lora_rank"), "lora_set_id": t.get("lora_set_id"), "lora_modules": t.get("lora_modules"),
                    "per_expert_rank": t.get("per_expert_rank")})
    return out


def confirmation_identity_mismatch(row: dict, candidate: dict, init_scheme: str | None) -> str:
    """Why a terminal trial at the picked (pair, lr) is NOT this pick's confirmation: its fixed axes must be the search's — the same clip constant
    (max_grad_norm), warm-up fraction and init scheme as the candidate / the design — and it must be a completed trial that passed the floor
    (status ok, floor_ok). Empty when it matches."""
    why = []
    for k in ("max_grad_norm", "warmup_frac"):
        if row.get(k) is None or candidate.get(k) is None or not math.isclose(float(row[k]), float(candidate[k]), rel_tol=1e-9):
            why.append(f"{k} {row.get(k)!r} vs the pick's {candidate.get(k)!r}")
    if init_scheme and row.get("init_scheme") and row["init_scheme"] != init_scheme:
        why.append(f"init_scheme {row['init_scheme']!r} vs the design's {init_scheme!r}")
    if row.get("status") != "ok":
        why.append(f"status {row.get('status')!r}")
    if row.get("floor_ok") is False:
        why.append("floor_ok false")
    return "; ".join(why)


def confirmation_record(search_dir: Path, key, confirming: list[dict], candidate: dict, init_scheme: str | None) -> dict | None:
    """The trial a promotion rests on, recorded as the pick's CONFIRMATION: the best completed trial at the promoted coordinates whose identity
    matches the pick (a round-9 confirmation, a confirm-pick trial, or a completed ADDENDUM row at the pair: completed work is canonical, a
    fresh confirm trial is never re-run over it). None when no confirming trial matches; a confirming trial
    whose fixed axes differ from the pick's refuses by name (it is another protocol's evidence, not this search's confirmation)."""
    if not confirming:
        return None
    y_key = "best_heldout_loss"
    rows = {r["trial_id"]: r for r in addendum_rows_at(search_dir, key, y_key)}
    best = min(confirming, key=lambda t: float(t["y"]))
    role = best.get("trial_role") or ("confirm_pick" if best.get("confirm_pick") else "confirmation" if best.get("confirmation") else None)
    row = rows.get(best["trial_id"])
    if row is None:                                            # a round-9 confirmation / confirm-pick trial: the summary's own axes
        p = Path(search_dir) / best["trial_id"] / "summary.json"
        try:
            smry = json.loads(p.read_text()); t = smry.get("trial") or {}
        except (OSError, ValueError):
            smry, t = {}, {}
        row = {"trial_id": best["trial_id"], "max_grad_norm": t.get("max_grad_norm"), "warmup_frac": t.get("warmup_frac"),
               "init_scheme": t.get("init_scheme") or smry.get("init_scheme"), "status": smry.get("status"), "floor_ok": smry.get("floor_ok")}
    else:
        role = "addendum"
    why = confirmation_identity_mismatch(row, candidate, init_scheme)
    if why:
        raise SystemExit(f"[hpo-pick] REFUSED: the trial {best['trial_id']} at the picked coordinates (rank {key[0]} / set {key[1]}, lr "
                         f"{float(candidate['lr']):.4g}) is not this search's confirmation — {why}; nothing promoted")
    return {"trial_id": best["trial_id"], "trial_role": role, "role_at_pick": "confirmation", "lr": best.get("lr"), "y": best.get("y"),
            "clip_activation_rate": best.get("clip_activation_rate"), "clip_n_steps": best.get("clip_n_steps"),
            "identity": {k: row.get(k) for k in ("max_grad_norm", "warmup_frac", "init_scheme", "status", "floor_ok")}}


def clip_gate_for_pick(search: dict, key, candidate: dict) -> dict:
    """The clip gate on the PICKED pair and learning rate (FD 6): the
    measured clip-activation evidence at exactly those coordinates —
    every terminal trial of the search there: the confirmation, the
    exploration replicate, a CONFIRM-PICK trial — must show a completed
    trial below the 25 % limit and none at/over it. The candidate comes
    back with its ``clip_evidence`` block (threshold, the trials, verdict
    ``promotable`` | ``unconfirmed`` (no trial ran at the coordinates: an
    inferred learning rate the confirm-pick trial can confirm) |
    ``refused`` (a trial ran there and the regime is not promotable —
    over the limit, unmeasured, or failed), reason). do_pick acts on the
    verdict: promote / request the confirm-pick / refuse by name."""
    threshold = tune_arm.recon_clip.CLIP_ACTIVATION_MAX
    ev = tune_arm.trial_clip_evidence_at(Path(search["dir"]), key, candidate["lr"])
    ok, why = tune_arm.clip_gate(ev, threshold=threshold)
    verdict = "promotable" if ok else ("unconfirmed" if not ev else "refused")
    candidate["clip_evidence"] = {"threshold": threshold, "trials": ev,
                                  "verdict": verdict, "reason": why}
    return candidate


def lr_for_pair(search: dict, key) -> dict:
    """The learning rate the pick ships and where it came from: the
    pair-only refit when it exists and covers this pair (its fitted
    argmin, else its promoted measured trial), else the design fit's
    prediction at the pair (when PD and interior), else the pair's best
    measured trial. Nothing invented. EVERY path exits through
    :func:`clip_gate_for_pick`, which stamps the
    verdict; a caller that must not promote an unconfirmed or refused
    regime reads ``clip_evidence["verdict"]`` (do_pick does)."""
    return clip_gate_for_pick(search, key, _lr_candidate(search, key))


def refusal_message(key, candidate: dict) -> str:
    ce = candidate["clip_evidence"]
    return (f"[hpo-pick] REFUSED: rank {key[0]} / set {key[1]} at lr "
            f"{candidate['lr']:.4g} ({candidate['source']}) is not promotable — {ce['reason']}. "
            f"FD 6: clip activation over {ce['threshold']:.0%} is never promoted. The pick stays the "
            "user's — pick again (another pair, or a pair whose confirm-pick trial measured below the "
            "limit); nothing was promoted, the ruling for this pick stands as recorded")


def confirm_pick_request(search: dict, key, candidate: dict, *, model_key: str, ruling_id: str,
                         modules: list[str], per_expert_rank, init_scheme, index: int,
                         predecessor: dict | None = None) -> dict:
    """The request `03_hpo.sh --phase confirm-pick` submits: ONE trial at the picked pair at the
    candidate learning rate (coordinates verbatim) under the identity RESERVED for ``index``
    (:func:`confirm_pick_identity`: trial id, per-request design path, job-name suffix), naming the pending
    request it supersedes (``predecessor``) so 03 can refuse the submission while that trial is still active."""
    ident = confirm_pick_identity(index)
    return {"kind": "confirm-pick", "search_id": search["search_id"],
            "campaign": search["design"].get("campaign") or (search["trials"] or {}).get("campaign"),
            "model_key": model_key, "ruling_id": ruling_id, "design_fit": "r1_fit.json",
            "round": tune_arm.CONFIRM_PICK_ROUND, **ident,
            "request": confirm_pick_request_name(index), "predecessor": predecessor,
            "pair": [key[0], key[1]],
            "pick": {"lr": candidate["lr"], "warmup_frac": candidate["warmup_frac"],
                     "max_grad_norm": candidate["max_grad_norm"], "lora_rank": key[0],
                     "lora_set_id": key[1], "lora_modules": list(modules),
                     "per_expert_rank": per_expert_rank, "init_scheme": init_scheme},
            "lr_source": candidate["source"], "reason": candidate["clip_evidence"]["reason"],
            "requested_utc": utc_now(),
            "note": ("the user's pick at a pair with no measured clip-activation evidence at its learning "
                     "rate (FD 6a: no default, never a dead end): ONE trial there decides the promotion — "
                     "below the 25 % limit promotes, at/over refuses by name; 03_hpo.sh --phase confirm-pick "
                     "submits it, --phase pick re-runs the gate when its summary lands")}


def prior_pick(search_dir: Path) -> dict | None:
    p = Path(search_dir) / "hpo_pick.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def ruling_already_recorded(prior: dict | None, *, model_key: str, key) -> bool:
    """The phase-A ruling is recorded ONCE per distinct pick (model, rank, set): a re-run of the same pick
    (after its confirm-pick trial) does not re-add it; a different pick supersedes it (same id, the
    fold keeps the latest)."""
    return bool(prior and prior.get("ruling_recorded") and prior.get("model_key") == model_key
                and prior.get("rank") == key[0] and prior.get("set") == key[1])


def _lr_candidate(search: dict, key) -> dict:
    """The ungated candidate of :func:`lr_for_pair` (its precedence)."""
    refit = search["refit"]
    if refit is not None and refit.get("refit_pair") and \
            tuple(tune_arm.pair_key(*refit["refit_pair"])) == key:
        sel = refit["selection"]
        return {"lr": float(sel["lr"]), "warmup_frac": float(sel["warmup_frac"]),
                "max_grad_norm": float(sel["max_grad_norm"]),
                "source": f"refit:{sel['selected_from']}",
                "trial_id": (refit.get("selection_guard") or {}).get("selected_trial"),
                "artifact": "r1+2_fit_pair.json",
                "alarms": [guard_mod.alarm_code(a) for a in
                           (refit.get("selection_guard") or {}).get("alarms") or []]}
    art = search["design"]
    fit = art.get("fit") or {}
    pred = (fit.get("pair_predictions") or {}).get(f"{key[0]}:{key[1]}")
    if pred and pred.get("lr") is not None and fit.get("positive_definite") \
            and pred.get("interior"):
        return {"lr": float(pred["lr"]), "warmup_frac": float(art["warmup_frac"]),
                "max_grad_norm": float(art["fixed_axes"]["max_grad_norm"]),
                "source": "design_fit:fitted_argmin_at_pair", "trial_id": None,
                "artifact": "r1_fit.json",
                "alarms": [guard_mod.alarm_code(a) for a in
                           (art.get("selection_guard") or {}).get("alarms") or []]}
    # the WIDENED-MENU refit (hpo_table.MENU_FIT_NAME): the design's rows refit over the menu the addendum widened — its prediction at a pair
    # the design never drew is the learning rate the addendum trial at that pair ran at (emit-trials --addendum-from, the same provenance);
    # PD and interior, exactly as the design fit's prediction is admitted
    mf = search.get("menu_fit")
    if mf is not None:
        mfit = mf.get("fit") or {}
        pred = (mfit.get("pair_predictions") or {}).get(f"{key[0]}:{key[1]}")
        if pred and pred.get("lr") is not None and mfit.get("positive_definite") and pred.get("interior"):
            return {"lr": float(pred["lr"]), "warmup_frac": float(mf["warmup_frac"]),
                    "max_grad_norm": float(mf["fixed_axes"]["max_grad_norm"]),
                    "source": "menu_refit:fitted_argmin_at_pair", "trial_id": None,
                    "artifact": "r1_fit_menu.json",
                    "alarms": [guard_mod.alarm_code(a) for a in
                               (mf.get("selection_guard") or {}).get("alarms") or []]}
    rows = pair_rows(art, key)
    y_key = art.get("y_key") or "best_heldout_loss"
    # the fold's own promotable rule (below the clip-activation threshold, or
    # no metrics): tune_arm.clip_activation_summary, the one owner
    best = (tune_arm.clip_activation_summary(rows, y_key=y_key)["best_raw_promotable_row"]
            if rows else None)
    if best is None:
        # a completed ADDENDUM trial at the pair (the measured evidence at a pair the design never drew; it ran at the set's best round-1 lr
        # when the widened refit was not PD): its own coordinates are the fallback, the best promotable one (status ok, floor passed, clip
        # under the threshold or unmeasured)
        thr = tune_arm.recon_clip.CLIP_ACTIVATION_MAX
        adds = [r for r in addendum_rows_at(Path(search["dir"]), key, y_key)
                if r.get("status") == "ok" and r.get("floor_ok") is not False and r.get(y_key) is not None
                and (r.get("clip_activation_rate") is None or float(r["clip_activation_rate"]) < thr)]
        if adds:
            a_best = min(adds, key=lambda r: float(r[y_key]))
            return {"lr": float(a_best["lr"]), "warmup_frac": float(a_best["warmup_frac"]),
                    "max_grad_norm": float(a_best["max_grad_norm"]),
                    "source": "addendum:best_measured_trial_at_pair", "trial_id": a_best["trial_id"],
                    "artifact": "r4 summaries", "alarms": ["not_the_fitted_argmin"]}
        raise SystemExit(f"[hpo-pick] pair {key} has no promotable measured trial "
                         "(every trial at it clipped over the threshold, or none "
                         "ran) and no fitted interior argmin — nothing to ship")
    return {"lr": float(best["lr"]), "warmup_frac": float(best["warmup_frac"]),
            "max_grad_norm": float(best["max_grad_norm"]),
            "source": "design_fit:best_measured_trial_at_pair",
            "trial_id": guard_mod.trial_id(best), "artifact": "r1_fit.json",
            "alarms": ["not_the_fitted_argmin"]}


def compose_phase_a(*, model_key: str, rank: int, set_id: str, modules: list[str],
                    per_expert_rank, search_id: str | None, trial_id: str | None,
                    table_url: str | None, ruled_utc: str) -> list[str]:
    target = {"lora_rank": int(rank), "lora_modules": list(modules),
              "per_expert_rank": per_expert_rank, "fit_search_id": search_id,
              "trial_id": trial_id, "phase": "architecture_selection"}
    reason = (f"LoRA architecture for {model_key} picked by the user from the "
              f"published rank x module-set table: rank {rank}, set {set_id}, "
              f"per-expert rank {per_expert_rank}")
    return ["registry", "ruling", "add", "--round", "r2", "--action", "operating_point",
            "--scope-kind", "model",
            "--scope-json", json.dumps({"match": {"model_ids": [model_key]}}),
            "--target-json", json.dumps(target, sort_keys=True),
            "--reason", reason,
            "--source-ref", f"{table_url or 'table'}#{search_id or 'no-search-id'}",
            "--ruled-by", "user", "--ruled-utc", ruled_utc,
            "--provenance-kind", "user_direct",
            "--id", PHASE_A_ID.format(model=model_key)]


def compose_phase_b(*, model_key: str, object_ids: list[str], ruled_utc: str) -> list[str]:
    a_id = PHASE_A_ID.format(model=model_key)
    return ["registry", "ruling", "add", "--round", "r2", "--action", "operating_point",
            "--scope-kind", "object",
            "--scope-json", json.dumps({"match": {"model_ids": [model_key],
                                                  "object_ids": list(object_ids)}}),
            "--target-json", json.dumps({"binds": a_id, "phase": "object_binding"}),
            "--reason", "binds the user's architecture selection to the published LoRA objects",
            "--source-ref", a_id, "--ruled-by", "user", "--ruled-utc", ruled_utc,
            "--provenance-kind", "user_direct",
            "--id", PHASE_B_ID.format(model=model_key)]


def update_models_yaml(path: Path, model_key: str, *, rank: int, modules: list[str],
                       per_expert_rank, ruling_id: str, write: bool) -> str:
    """Rewrite ONLY the model's `lora_architecture:` flow mapping (a single
    line under the model's block), leaving every other line byte-identical.
    Returns the new line. Refuses a table without the model or without the
    line."""
    text = path.read_text()
    lines = text.split("\n")
    in_model = False
    model_re = re.compile(rf"^  {re.escape(model_key)}:\s*(#.*)?$")
    other_model_re = re.compile(r"^  [A-Za-z0-9_]+:\s*(#.*)?$")
    la_re = re.compile(r"^(\s+)lora_architecture:\s*\{.*\}\s*(#.*)?$")
    new_line = None
    for i, ln in enumerate(lines):
        if model_re.match(ln):
            in_model = True
            continue
        if in_model and other_model_re.match(ln):
            break
        if in_model:
            m = la_re.match(ln)
            if m:
                indent = m.group(1)
                comment = f"  {m.group(2)}" if m.group(2) else ""
                per = ("null" if per_expert_rank is None
                       else json.dumps(per_expert_rank, sort_keys=True))
                mods = "[" + ", ".join(modules) + "]"
                new_line = (f"{indent}lora_architecture: {{rank: {int(rank)}, modules: {mods}, "
                            f"per_expert_rank: {per}, selected_by: user_pick, "
                            f"selection_ruling_id: {ruling_id}}}{comment}")
                lines[i] = new_line
                break
    if new_line is None:
        raise SystemExit(f"[hpo-pick] {path}: no `lora_architecture: {{...}}` line under "
                         f"model {model_key!r} — refusing to guess where the pick goes")
    if write:
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text("\n".join(lines))
        tmp.replace(path)
    return new_line


REGISTRY_CLI_MODULE = "sisteer.registry.cli"      # pyproject's `registry` console script (sisteer.registry.cli:main)


def registry_cli_argv(argv: list[str]) -> tuple[list[str], str]:
    """The argv that EXECUTES a composed `registry …` line: the `registry` console script when PATH has one, else the repo's own documented
    interim `python -m sisteer.registry.cli` (pyproject.toml: "until the venv is re-synced") on THIS interpreter — a venv without the
    console script installed is how the first live pick died (FileNotFoundError: 'registry', 2026-10-01). Returns (argv, how)."""
    if not argv or argv[0] != "registry":
        return list(argv), "verbatim"
    if shutil.which("registry"):
        return list(argv), "console script"
    return [sys.executable, "-m", REGISTRY_CLI_MODULE, *argv[1:]], f"{REGISTRY_CLI_MODULE} (no `registry` on PATH)"


def _registry_env() -> dict:
    """The exec's environment: the repo's src on PYTHONPATH so `-m sisteer.registry.cli` resolves to THIS checkout's code."""
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2] / "src")
    parts = [x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x]
    if src not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src, *parts])
    return env


def manual_selection_guard(search_dir: Path, confirmation: dict | None, *, ruling_id: str, search_id: str, menu: dict | None,
                           lr_src: dict) -> dict:
    """The fold row's provenance for the operator's pick — the shape hpo_selection_guard writes for a searched row
    (selected_trial / artefact / artefact_sha256 / fit_fingerprint), with the pick's own facts: basis user_pick, the ruling it was recorded
    under, the closed search, the menu identity the authorization read, and the confirming trial it rests on (its summary.json on the
    record is the artefact, hashed; its fit_fingerprint names the fitted object). A reader holds a manual row to the same bar as a
    searched one instead of exempting it."""
    g: dict = {"basis": "user_pick", "selected_from": "confirmation_trial" if confirmation else "menu_refit", "ruling_id": ruling_id,
               "search_id": search_id, "search_closed": True, "menu_sha256": (menu or {}).get("sha256"), "lr_source": lr_src.get("source"),
               "selected_trial": None, "trial_role": None, "artefact": None, "artefact_sha256": None, "fit_fingerprint": None, "selected_y": None}
    if confirmation:
        tid = str(confirmation["trial_id"]); sp = Path(search_dir) / tid / "summary.json"
        raw = sp.read_bytes(); summ = json.loads(raw)
        g.update({"selected_trial": tid, "trial_role": confirmation.get("trial_role"), "artefact": f"{tid}/summary.json",
                  "artefact_sha256": hashlib.sha256(raw).hexdigest(), "fit_fingerprint": summ.get("fit_fingerprint"),
                  "selected_y": confirmation.get("y")})
    return g


def _run_registry(argv: list[str], *, execute: bool) -> None:
    line = " ".join(shlex.quote(a) for a in argv)
    if not execute:
        print(f"DRY-RUN: {line}")
        return
    run_argv, how = registry_cli_argv(argv)
    print(f"[hpo-pick] running: {line}" + ("" if how == "console script" else f"  (via {how})"))
    r = subprocess.run(run_argv, text=True, capture_output=True, env=_registry_env())
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"[hpo-pick] `registry ruling add` failed (rc {r.returncode})")


def opening_config_sha256(opening: dict, fit_key: str, repo: Path) -> tuple[str | None, str | None]:
    """At the pick's close: the LoRA fit_search opening recorded the config's 16-hex prefix (its identity key), but the registry's
    fit_search.config_sha256 is a 64-hex sha256 or `unverified`. Returns ``(value, note)``: the FULL sha resolved from the config blob at the
    opening's code_sha — `git show <code_sha>:configs/sdfbase_<fit_key>_fits.yaml`, the bytes the opening hashed — verified against the prefix
    through tune_arm.close_config_sha256; a 64-hex opening as it is; an opening with no value -> (None, None) (the column is nullable). A
    READABLE blob that disagrees with the prefix REFUSES by name (a code_sha at another folded config, a wrong fit key, an altered opening —
    the close would register the search under another config identity); a missing or unavailable historical blob is a distinct, declared
    outcome: ("unverified", <the reason>), recorded with the close — never silent."""
    cur = str(opening.get("config_sha256") or "").strip().lower()
    if not cur:
        return None, None
    if cur == "unverified":
        return "unverified", "the opening recorded `unverified` (no config file sha available to the producer)"
    if re.fullmatch(r"[0-9a-f]{64}", cur):
        return cur, None
    code_sha = str(opening.get("code_sha") or "").strip()
    rel = f"configs/sdfbase_{fit_key}_fits.yaml"
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        note = f"the opening's config_sha256 {cur!r} is a 16-hex prefix and its code_sha {code_sha!r} names no commit — the full sha cannot be resolved; recorded as unverified"
        print(f"[registry] {note}")
        return "unverified", note
    r = subprocess.run(["git", "-C", str(repo), "show", f"{code_sha}:{rel}"], capture_output=True)
    if r.returncode != 0:
        note = (f"the opening's config_sha256 {cur!r} is a 16-hex prefix and no config blob {rel} is readable at its code_sha {code_sha[:12]} "
                f"({r.stderr.decode(errors='replace').strip()[:120]}) — the full sha cannot be resolved; recorded as unverified")
        print(f"[registry] {note}")
        return "unverified", note
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as fh:
        fh.write(r.stdout); tmp = fh.name
    try:
        full = tune_arm.close_config_sha256(dict(opening), tmp)
    except SystemExit as e:
        raise SystemExit(f"[hpo-pick] REFUSED: the config blob {rel} at the opening's code_sha {code_sha[:12]} does not hash to the opening's "
                         f"config_sha256 prefix {cur} ({e}) — the search's asserted config identity is inconsistent (a code_sha at another "
                         f"folded config, a wrong fit key, or an altered opening); the close would register the fit_search under another config, so "
                         f"nothing is promoted or closed until it is explained") from None
    finally:
        os.unlink(tmp)
    print(f"[registry] fit_search.config_sha256 resolved from the config blob at the opening's code_sha {code_sha[:12]}: {full}")
    return full, None



def _close_search(search: dict, pick: dict, *, execute: bool) -> None:
    sid = search["search_id"]
    if not sid:
        print("[registry] the search carries no search_id — fit_search not closed")
        return
    if not execute:
        print(f"DRY-RUN: emit_hpo.close_search({sid!r}, winner_params={json.dumps(pick['winner_params'], sort_keys=True)})")
        return
    eh = tune_arm._emit_hpo()
    if eh is None:
        return
    art = search["refit"] or search["design"]
    fit = art.get("fit") or {}
    fu = art.get("followup") or {}
    campaign = search["design"].get("campaign") or (search["trials"] or {}).get("campaign")
    opening = search["design"].get("search_opening") or (search["trials"] or {}).get("search_opening")
    if not campaign:
        print("[registry] the search names no campaign — fit_search not closed")
        return
    # every terminal trial of the search across every round
    # (design + refine + the confirmation / exploration replicate), reconciled against the emitted
    # fit_trial rows — the design artefact's rows are the fit's inputs, not the search's trials
    done = tune_arm.reconcile_completed_trials(Path(search["dir"]), str(sid), campaign=str(campaign))   # evidence under another campaign holds the close
    if not opening:
        raise SystemExit(f"[registry] neither the design fit nor r1_trials.json carries the "
                         f"search_opening payload for {sid} — refusing to close the fit_search "
                         "with a hollow superseding record (the fold keeps the latest one)")
    opening = dict(opening)
    res = pick.get("fit_search_config_sha256") or {}
    if res.get("value") is not None:                                 # resolved BEFORE the promotion's writes (do_pick); a disagreeing blob refused there
        opening["config_sha256"] = res["value"]
    # the emitter needs a trial id and an objective, never None: the pick's lr source names its trial when it
    # promoted one; a fitted argmin falls back to the best measured trial at the pair as the nearest evidence
    tid = pick.get("winner_trial_id") or pick["lr_source"].get("trial_id")
    obj = pick.get("winner_objective")
    # the follow-up's within-noise verdict was measured on the round-9 confirmation against the OLD pair's fitted
    # minimum (tune_arm._followup_evidence); it is the recorded confirmation's only when that is the trial it measured — else null, with its
    # provenance in extra, never a verdict attributed to an addendum trial it never saw
    conf_tid = (pick.get("confirmation") or {}).get("trial_id") or fu.get("confirmation_trial_id")
    conf_noise = fu.get("confirmation_within_noise") if (conf_tid and fu.get("confirmation_trial_id") == conf_tid) else None
    noise_prov = ({"measured_on": fu.get("confirmation_trial_id"), "attributed": conf_noise is not None,
                   "note": None if conf_noise is not None else "the follow-up's within-noise verdict was measured on another trial (or none); not attributed to this confirmation"}
                  if fu.get("confirmation_trial_id") or conf_tid else None)
    if tid is None or obj is None:
        y_key = search["design"].get("y_key") or "best_heldout_loss"
        rows = pair_rows(search["design"], tune_arm.pair_key(pick["rank"], pick["set"])) or search["design"].get("rows") or []
        best = min(rows, key=lambda r: float(r[y_key]))
        tid = tid or guard_mod.trial_id(best)
        obj = obj if obj is not None else float(best[y_key])
    eh.close_search(
        sid, campaign=str(campaign), opening=dict(opening),
        winner_params=pick["winner_params"], winner_trial_id=str(tid), winner_objective=float(obj),
        argmin_interior=pick["lr_source"]["source"].endswith("fitted_argmin")
        or pick["lr_source"]["source"].endswith("fitted_argmin_at_pair"),
        # note: a design-fit argmin at an off-winner pair is interior in u but confirmed by its own trial
        quadratic_fit_positive_definite=bool(fit.get("positive_definite")),
        quadratic_fit_params=fit.get("coef_by_name"),
        recentre_rounds=max(len(art.get("rounds") or [1]) - 1, 0),
        confirmation_trial_id=conf_tid,
        confirmation_within_noise=conf_noise,
        noise_estimate=fit.get("sigma_log_y"),
        grad_norm_reconnaissance_uri=(search["design"].get("recon") or {}).get("path"),
        trials_uri=pick.get("table_url"),
        n_trials_completed=done["n"],
        extra={"selected_by": "user_pick", "architecture_selection_ruling_id": pick["ruling_id"],
               "pick": pick["winner_params"], "lr_source": pick["lr_source"],
               "alarms_at_pick": pick["lr_source"].get("alarms"),
               "clip_evidence_at_pick": pick["lr_source"].get("clip_evidence"),
               "confirmation": pick.get("confirmation"), "confirmation_noise_provenance": noise_prov,
               "config_sha256_resolution": (pick.get("fit_search_config_sha256") or {}).get("note"),
               "menu_at_pick": pick.get("menu"),
               "trials_completed": done["trial_ids"], "n_trials_by_status": done["by_status"],
               "exploration_replicate": fu.get("exploration_replicate")})
    print(f"registry: fit_search {sid} (closed with the user's pick)")


def do_pick(a) -> int:
    search_dir = Path(a.search_dir)
    search = load_search(search_dir, a.allowed_pairs, getattr(a, "source_record", None))
    allowed = search["allowed"]
    set_id = str(a.set)
    pr = tune_arm.refuse_pair_outside(allowed, a.rank, set_id, who="hpo-pick")
    key = tune_arm.pair_key(pr["rank"], pr["set"])
    modules = list(pr["lora_modules"])
    per = pr.get("per_expert_rank")
    if per is None:
        # a dense model: no experts; the search rows may still carry the map
        rows = pair_rows(search["design"], key)
        per = next((r.get("per_expert_rank") for r in rows if r.get("per_expert_rank")), None)
    # the models.yaml key: --model-key, else the probe's model_id without the fit
    # slug's `_base`; never the HF repo id, never a minted placeholder
    model_key = a.model_key or (search["allowed"].get("model_id") or "").replace("_base", "")
    if not model_key:
        raise SystemExit("[hpo-pick] --model-key is required: the allowed-pairs document "
                         "names no model_id, and the ruling id / models.yaml row are keyed "
                         "by it (no default is minted)")
    lr_src = lr_for_pair(search, key)
    ce = lr_src["clip_evidence"]
    ruled_utc = a.ruled_utc or utc_now()
    ruling_id = PHASE_A_ID.format(model=model_key)
    argv = compose_phase_a(model_key=model_key, rank=key[0], set_id=key[1], modules=modules,
                           per_expert_rank=per, search_id=search["search_id"],
                           trial_id=lr_src.get("trial_id"), table_url=a.table_url,
                           ruled_utc=ruled_utc)
    init_scheme = (search["design"].get("init_scheme")
                   or (search["trials"].get("init_scheme") if search["trials"] else None))
    fit_key = a.fit_key or (search["allowed"].get("model_id") or f"{model_key}_base")
    print(f"[hpo-pick] pick: rank {key[0]} set {key[1]} modules {modules} per_expert_rank "
          f"{per}; lr {lr_src['lr']:.4g} ({lr_src['source']}); alarms at the pick: "
          f"{lr_src.get('alarms') or 'none'}; clip gate: {ce['verdict']} — {ce['reason']}")
    # FD 6a: the pick is the operator's and is recorded EITHER WAY (the ruling, user_direct), once per distinct pick
    prior = prior_pick(search_dir)
    if ruling_already_recorded(prior, model_key=model_key, key=key):
        print(f"registry: ruling {ruling_id} (already recorded for rank {key[0]} / set {key[1]})")
        recorded = True
    else:
        _run_registry(argv, execute=a.execute)
        print(f"registry: ruling {ruling_id}")
        recorded = bool(a.execute)
    base = {"model_key": model_key, "fit_key": fit_key, "rank": key[0], "set": key[1],
            "lora_modules": modules, "per_expert_rank": per, "init_scheme": init_scheme,
            "search_id": search["search_id"], "ruling_id": ruling_id, "ruled_utc": ruled_utc,
            "lr_source": lr_src, "ruling_argv": argv, "ruling_recorded": recorded,
            "table_url": a.table_url, "executed": bool(a.execute)}
    out = search_dir / "hpo_pick.json"
    if ce["verdict"] == "refused":
        # a trial ran at these coordinates and the regime is not promotable: refuse by name with the measured
        # rate; the pick record says so, nothing is promoted, the operator picks again
        if a.execute:
            out.write_text(json.dumps({**base, "promoted": False, "refused": ce["reason"]},
                                      indent=2, sort_keys=True, default=str) + "\n")
        raise SystemExit(refusal_message(key, lr_src))
    if ce["verdict"] == "unconfirmed":
        # no trial at these coordinates: not promoted — request the confirm-pick trial (an off-winner pick is
        # never a dead end); an identical pending request is left as it is
        req_path = search_dir / CONFIRM_PICK_REQUEST_NAME
        pending = None
        if req_path.is_file():
            try:
                pending = json.loads(req_path.read_text())
            except (OSError, ValueError):
                pending = None
        # the same pick stands only when the pending request already RESERVED an identity (a pre-round-2 pointer with no
        # index / request file is superseded by a proper reservation, never emitted from)
        same = bool(pending and pending.get("pair") == [key[0], key[1]]
                    and pending.get("pick", {}).get("lr") == lr_src["lr"]
                    and isinstance(pending.get("index"), int) and pending.get("request"))
        if same:
            req = pending
        else:
            # the index is RESERVED here, one past every request
            # already in the ledger / every landed r8 trial / the pending pointer — never a count of fetched trial
            # dirs — and the request is written to its own immutable per-index file before the pointer moves, so a
            # queued predecessor keeps its design and its trial dir whatever is picked next
            index = next_confirm_pick_index(search_dir, pending)
            predecessor = None
            if pending and isinstance(pending.get("index"), int):
                p_id = confirm_pick_identity(pending["index"])
                predecessor = {**p_id, "request": pending.get("request") or confirm_pick_request_name(pending["index"]),
                               "pair": pending.get("pair"), "lr": (pending.get("pick") or {}).get("lr"),
                               "landed": (search_dir / p_id["trial_id"] / "summary.json").is_file()}
            req = confirm_pick_request(search, key, lr_src, model_key=model_key, ruling_id=ruling_id,
                                       modules=modules, per_expert_rank=per, init_scheme=init_scheme,
                                       index=index, predecessor=predecessor)
        ledger_path = search_dir / req["request"]
        if a.execute:
            if same:
                print(f"[hpo-pick] confirm-pick already requested for rank {key[0]} / set {key[1]} at lr "
                      f"{lr_src['lr']:.4g} ({req_path}; index {req.get('index')}, trial {req.get('trial_id')}, "
                      f"design {req.get('design')}) — waiting for its trial")
            else:
                if ledger_path.exists():
                    raise SystemExit(f"[hpo-pick] REFUSED: {ledger_path} already exists — confirm-pick index "
                                     f"{req['index']} is reserved by an earlier request and a reserved request is "
                                     "immutable; the ledger disagrees with the pointer (edited by hand?) — repair it")
                text = json.dumps(req, indent=2, sort_keys=True, default=str) + "\n"
                ledger_path.write_text(text)                       # the immutable reservation first …
                req_path.write_text(text)                          # … then the pointer 03 reads
                print(f"[hpo-pick] wrote {ledger_path} (reserved index {req['index']}: trial {req['trial_id']}, "
                      f"design {req['design']}, job suffix {req['job_suffix']}) and the pointer {req_path}")
                if req.get("predecessor"):
                    pr = req["predecessor"]
                    print(f"[hpo-pick] supersedes the pending request t{pr['index']:02d} (rank {pr['pair'][0]} / set "
                          f"{pr['pair'][1]} at lr {pr['lr']:.4g}; its trial {pr['trial_id']} "
                          f"{'has landed' if pr['landed'] else 'has NOT landed'}) — 03_hpo.sh --phase confirm-pick "
                          "refuses to submit this one while that trial is still queued/running")
            out.write_text(json.dumps({**base, "promoted": False,
                                       "confirm_pick": {"request": str(req_path), "request_file": str(ledger_path),
                                                        "index": req.get("index"), "trial_id": req.get("trial_id"),
                                                        "design": req.get("design"), "job_suffix": req.get("job_suffix"),
                                                        "predecessor": req.get("predecessor"),
                                                        "lr": req["pick"]["lr"],
                                                        "requested_utc": req.get("requested_utc")}},
                                      indent=2, sort_keys=True, default=str) + "\n")
        else:
            print(f"DRY-RUN: write {ledger_path} and {req_path}: {json.dumps(req['pick'], sort_keys=True)}")
        print(f"[hpo-pick] CONFIRM-PICK requested: rank {key[0]} / set {key[1]} at lr {lr_src['lr']:.4g} "
              f"({lr_src['source']}) has no measured clip-activation evidence — NOT promoted (the ruling "
              f"stands); `03_hpo.sh --phase confirm-pick` submits ONE trial there (round "
              f"{tune_arm.CONFIRM_PICK_ROUND}), then re-run `--phase pick --rank {key[0]} --set {key[1]}`: "
              "below the 25% limit promotes, at/over refuses by name")
        return 0
    # promotable: the learning rate ships. The trial the promotion rests on is recorded as the pick's CONFIRMATION — a round-9 confirmation,
    # a confirm-pick trial, or a completed ADDENDUM row at the pair (completed work is canonical, no
    # confirm trial is re-run over it); its fixed axes must be the search's, else refused by name (confirmation_record)
    confirming = [t for t in ce["trials"] if t.get("status") == "ok" and t.get("y") is not None]
    confirmation = confirmation_record(search_dir, key, confirming, lr_src, init_scheme)
    if confirmation is not None:
        print(f"[hpo-pick] the completed {confirmation['trial_role']} trial {confirmation['trial_id']} at these coordinates IS the confirmation "
              f"(status ok, floor passed, clip activation {confirmation.get('clip_activation_rate')}) — no confirm-pick trial is requested")
    fold_entry = {"lr": lr_src["lr"], "warmup_frac": lr_src["warmup_frac"],
                  "max_grad_norm": lr_src["max_grad_norm"], "lora_rank": key[0],
                  "lora_modules": modules, "per_expert_rank": per,
                  "init_scheme": init_scheme, "fit_search_id": search["search_id"],
                  "architecture_selection_ruling_id": ruling_id,
                  "fit_lr_selected_by": "manual",
                  # the manual row carries its own provenance guard (the searched rows carry hpo_selection_guard's)
                  "selection_guard": manual_selection_guard(search_dir, confirmation, ruling_id=ruling_id, search_id=search["search_id"],
                                                            menu=allowed, lr_src=lr_src)}
    winner_params = {"lr": lr_src["lr"], "warmup_frac": lr_src["warmup_frac"],
                     "max_grad_norm": lr_src["max_grad_norm"], "lora_rank": key[0],
                     "lora_modules": modules, "lora_set_id": key[1],
                     "per_expert_rank": per, "init_scheme": init_scheme}
    y_key = search["design"].get("y_key") or "best_heldout_loss"
    win_obj = None
    win_tid = lr_src.get("trial_id")
    if win_tid:
        art = search["refit"] or search["design"]
        row = next((r for r in art.get("rows") or []
                    if guard_mod.trial_id(r) == win_tid), None)
        win_obj = float(row[y_key]) if row else None
    elif confirming:
        # an INFERRED learning rate promoted through the trial that measured it there (the round-9
        # confirmation, or the confirm-pick trial): the promoted point is one that actually ran
        best = min(confirming, key=lambda t: float(t["y"]))
        win_tid, win_obj = best["trial_id"], float(best["y"])
    elif (search["refit"] or {}).get("selection", {}).get("fitted_min_log_y") is not None \
            and tuple(tune_arm.pair_key(*(search["refit"].get("refit_pair") or (None, None)))) == key:
        win_obj = math.exp(search["refit"]["selection"]["fitted_min_log_y"])
    # the menu the pick was chosen from (a widened or replaced menu after the promotion HOLDS it — a changed menu)
    menu_rec = {"path": allowed.get("path"), "sha256": allowed.get("sha256"), "result_generation": allowed.get("result_generation"),
                "n_pairs": len(allowed.get("pairs") or [])}
    # the fit_search's config identity, resolved BEFORE anything is written: a disagreeing blob refuses here, nothing promoted
    opening0 = search["design"].get("search_opening") or (search["trials"] or {}).get("search_opening")
    cfg_value, cfg_note = opening_config_sha256(dict(opening0), fit_key, HERE.parent.parent) if opening0 else (None, None)
    pick = {**base, "promoted": True, "winner_params": winner_params, "winner_objective": win_obj,
            "winner_trial_id": win_tid, "fold": {fit_key: {ARM: fold_entry}},
            "alarms_at_pick": lr_src.get("alarms"), "confirmation": confirmation, "menu": menu_rec,
            "fit_search_config_sha256": {"value": cfg_value, "note": cfg_note}}
    pick = annotate_hold_cleared(search_dir, pick)                  # the hold it clears is recorded first; the FILE moves only after every write (below)
    if a.models_table:
        line = update_models_yaml(Path(a.models_table), model_key, rank=key[0], modules=modules,
                                  per_expert_rank=per, ruling_id=ruling_id, write=a.execute)
        print(f"{'' if a.execute else 'DRY-RUN: '}models.yaml {model_key}: {line.strip()}")
    out = search_dir / "hpo_pick.json"
    fold_out = search_dir / "hpo_fold.json"
    if a.execute:
        out.write_text(json.dumps(pick, indent=2, sort_keys=True, default=str) + "\n")
        fold_out.write_text(json.dumps(pick["fold"], indent=2, sort_keys=True) + "\n")
        print(f"[hpo-pick] wrote {out} and {fold_out} (the --fold-hpo input)")
    else:
        print(f"DRY-RUN: write {out} and {fold_out}: {json.dumps(pick['fold'], sort_keys=True)}")
    if a.fit_config:
        cmd = [sys.executable, str(HERE / "make_sdfbase_configs.py"), "--models", fit_key,
               "--fold-hpo", str(fold_out)]
        print(("" if a.execute else "DRY-RUN: ") + "regenerate the fit config: "
              + " ".join(shlex.quote(c) for c in cmd) + f"  # -> {a.fit_config}")
    _close_search(search, pick, execute=a.execute)
    finalize_hold_clear(search_dir, execute=a.execute)             # the last step: every promotion write and the close succeeded
    return 0


def do_bind(a) -> int:
    ids = [x for x in str(a.object_ids or "").split(",") if x.strip()]
    if len(ids) != 3:
        raise SystemExit("[hpo-pick] --bind needs --object-ids with EXACTLY three shas "
                         "(control LoRA, constitution LoRA, difference LoRA)")
    search_dir = Path(a.search_dir)
    pick_p = search_dir / "hpo_pick.json"
    if not pick_p.is_file():
        raise SystemExit(f"[hpo-pick] --bind needs phase A first ({pick_p} absent)")
    pick = _load_json(pick_p)
    model_key = a.model_key or pick["model_key"]
    argv = compose_phase_b(model_key=model_key, object_ids=ids, ruled_utc=a.ruled_utc or utc_now())
    _run_registry(argv, execute=a.execute)
    b_id = PHASE_B_ID.format(model=model_key)
    print(f"registry: ruling {b_id}")
    if a.execute:
        pick["phase_b"] = {"ruling_id": b_id, "object_ids": ids, "argv": argv, "utc": utc_now()}
        pick_p.write_text(json.dumps(pick, indent=2, sort_keys=True, default=str) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--search-dir", required=True)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--set", default=None, help="module set id: wo|qkvo|attn_mlp|mlp|down")
    ap.add_argument("--allowed-pairs", default=None)
    ap.add_argument("--models-table", default=None, help="reproduce/models.yaml to update")
    ap.add_argument("--model-key", default=None, help="the models.yaml key (default: the "
                    "probe's model_id without `_base`)")
    ap.add_argument("--fit-key", default=None, help="the fit config key (default: the probe's model_id)")
    ap.add_argument("--fit-config", default=None, help="configs/sdfbase_<key>_fits.yaml to regenerate")
    ap.add_argument("--table-url", default=None)
    ap.add_argument("--ruled-utc", default=None)
    ap.add_argument("--bind", action="store_true", help="phase B: bind phase A to the three objects")
    ap.add_argument("--object-ids", default=None, help="phase B: three comma-separated object shas")
    ap.add_argument("--source-record", default=None, help="<cluster>:<record dir> of the search (the widened-menu refit's binding, as hpo_table checks it)")
    ap.add_argument("--verify-promotion", action="store_true",
                    help="READ-ONLY: re-evaluate the recorded promotion (hpo_pick.json) against the current evidence — every fit bound, the recorded (pair, lr) "
                         "still selected, the clip gate at the promoted coordinates still promotable; exit 0 iff it stands (D543; the review on d464ad584)")
    ap.add_argument("--write-hold", default=None, metavar="REASON",
                    help="D543: write hpo_pick.HOLD.json beside the search (the pick phase found the promoted pick's basis changed) and exit")
    ap.add_argument("--execute", action="store_true", help="run `registry ruling add`, write "
                    "models.yaml / hpo_pick.json, close the search (default: print everything)")
    a = ap.parse_args(argv)
    if a.verify_promotion:
        reasons = verify_promotion(Path(a.search_dir), a.allowed_pairs, a.source_record)
        if reasons:
            print("[hpo-pick] the recorded promotion no longer stands: " + "; ".join(reasons), flush=True)
            return 1
        print("[hpo-pick] the recorded promotion stands over the current evidence (fits bound, (pair, lr) selected, clip gate promotable)", flush=True)
        return 0
    if a.write_hold:
        p = write_hold(Path(a.search_dir), a.write_hold)
        print(f"[hpo-pick] HOLD written: {p} — every consumer of hpo_pick.json / the folded config refuses by name while it stands")
        return 0
    if a.bind:
        return do_bind(a)
    if a.rank is None or not a.set:
        raise SystemExit("[hpo-pick] --rank and --set are REQUIRED: the pick is the user's "
                         "(FD 6a) — no default is applied in their absence")
    return do_pick(a)


if __name__ == "__main__":
    raise SystemExit(main())
