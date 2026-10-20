#!/usr/bin/env python
"""hpo_table — the published per-trial table of a LoRA architecture search
and of the ReLU search.

    python scripts/sdfbase/hpo_table.py --search-dir <hpo root> \
        [--allowed-pairs allowed_pairs.json] --out <dir>

Reads the search directory `tune_arm.py trial` / `analyze --design v2` wrote
— `r<R>_t<NN>/summary.json` for every round (the design round, the refine
round, the round-9 confirmation, the round-3 exploration replicate), the
design fit `r1_fit.json`, the pair-only refit `r1+2_fit_pair.json` — and
renders `table.csv`, `table.md` and `table.html` in --out:

  * one row per trial: trial_id, rank, set, per_expert_rank,
    trainable_params, lr, warmup, clip, best_heldout_CE, epochs,
    clip_fraction, status, alarms (the guard's alarms VERBATIM on the
    selected trial; the exploration replicate marked as such, never as a
    confirmation);
  * a rank x module-set pivot of the FITTED beta+gamma prediction at u* over
    the ALLOWED pairs only (lower is better); an excluded pair is shown
    greyed with the probe's reason, never as a candidate; every prediction
    is marked model-dependent when the additive model is inadequate;
  * the model-adequacy line and the alarms, verbatim.

The user picks from this table (FD 6a); nothing here applies a default.
Pure python + the sibling tune_arm/hpo_selection_guard modules loaded by
path (tune_arm imports torch only inside its GPU-side functions, so no torch
is needed to render); trainable_params is read off the
summaries (the built object's count) and, before a trial has run, off the
probe's allowed_pairs.json.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent


#: the round-1 rows refit over a widened menu, written by `03 --phase addendum` beside the design fit (never over it)
MENU_FIT_NAME = "r1_fit_menu.json"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ONE tune_arm instance per process (the tests register theirs as `_tune_arm`
# — hpo_next_round's convention); the guard rides on it, so the table reads
# the alarms with the module that filed them.
tune_arm = (sys.modules.get("_tune_arm") or sys.modules.get("tune_arm")
            or _load_by_path("tune_arm", SCRIPTS / "tune_arm.py"))
guard_mod = tune_arm.hpo_selection_guard

COLUMNS = ("trial_id", "round", "role", "rank", "set", "per_expert_rank",
           "trainable_params", "lr", "warmup", "clip", "best_heldout_CE",
           "epochs", "clip_fraction", "status", "alarms")


def _fmt(v, spec=".4g") -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return format(v, spec)
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)          # bool / int / str


def _fit_identity(fit: dict) -> list[tuple[str, float]]:
    """(trial id, y) of the rows a v2 fit was fit over: row_identity when recorded, else from its rows with its own y_key (the design fit on the
    live record predates row_identity)."""
    if fit.get("row_identity"):
        return [tuple(x) for x in fit["row_identity"]]
    yk = fit.get("y_key") or "best_heldout_loss"
    return sorted((tune_arm.hpo_selection_guard.trial_id(r), float(r[yk])) for r in (fit.get("rows") or []) if r.get(yk) is not None)


def _fit_rows_binding(fit: dict, summaries: list[dict]) -> str | None:
    """Why a v2 fit may NOT stand over the CURRENT summaries (None = bound). A retried trial that replaced its summary after
    the fit changes the row the table lists while the pivot and the winner would come from the stale fit — the two would be published together
    as one table. So: every row the fit was fit over must be a current usable summary of its rounds with the SAME objective value, and every
    current usable summary of those rounds must be a fit row or one of the fit's recorded exclusions. Keyed on the artefact KIND (every
    `analyze --design v2` writes rows + excluded), not on a field's presence."""
    if fit.get("design") != "v2":
        return None
    yk = fit.get("y_key") or "best_heldout_loss"
    rounds = {int(r) for r in (fit.get("rounds") or [])}
    ident = dict(_fit_identity(fit))
    # An exclusion is a fact about the summary AS IT WAS at fit time. The only exclusion that stays valid for a summary that
    # is usable now is the pair-only refit's deliberate "other_pair"; a status / floor / non-finite exclusion whose summary has since become
    # usable (a retried trial at the same path) is a row the fit never saw, not an exemption.
    excluded = {str(e[0]): (str(e[1]) if len(e) > 1 else "") for e in (fit.get("excluded") or []) if e}
    INTENTIONAL = {"other_pair"}
    current: dict[str, tuple] = {}
    for s in summaries:
        tid = s["_trial_id"]; m = re.match(r"r(\d+)_t\d+$", tid)
        rnd = int(s["round"]) if s.get("round") is not None else (int(m.group(1)) if m else None)
        if rnd not in rounds:
            continue
        y = s.get(yk)
        yf = float(y) if y is not None else None
        usable = s.get("status") == "ok" and yf is not None and math.isfinite(yf) and s.get("floor_ok") is not False
        current[tid] = (usable, yf, s.get("_clip_fraction"), int(s.get("_clip_n_steps") or 0))
    bad = []
    for tid, y in sorted(ident.items()):
        cur = current.get(tid)
        if cur is None:
            bad.append(f"{tid}: a fit row with no summary on the mirror")
        elif not cur[0]:
            bad.append(f"{tid}: a fit row whose summary is now unusable")
        elif cur[1] != y:
            bad.append(f"{tid}: fit {y!r}, current {cur[1]!r}")
    for tid, cur in sorted(current.items()):
        usable = cur[0]
        if not usable or tid in ident:
            continue
        if tid not in excluded:
            bad.append(f"{tid}: a current usable summary the fit never saw")
        elif excluded[tid] not in INTENTIONAL:
            bad.append(f"{tid}: excluded at fit time ({excluded[tid]}), now a usable summary")
    # The fit's selection, exclusions and alarms were computed from the clip evidence its rows carried (clip_activation_rate /
    # clip_n_steps, read from each trial's metrics.jsonl at fit time). A sidecar backfilled or re-run since (same summary, other metrics) moves the
    # table key and the per-trial clip fraction, but not the fit's conclusions — the fit is bound to the evidence it saw, value for value.
    def _clip_txt(rate, n):
        return "no clip evidence" if rate is None else f"clip activation {float(rate):.3f} over {n} steps"
    for r in (fit.get("rows") or []):
        tid = tune_arm.hpo_selection_guard.trial_id(r); cur = current.get(tid)
        if cur is None or not cur[0] or "clip_activation_rate" not in r:
            continue
        f_rate, f_n = r.get("clip_activation_rate"), int(r.get("clip_n_steps") or 0)
        c_rate, c_n = cur[2], cur[3]
        same_rate = (f_rate is None and c_rate is None) or (f_rate is not None and c_rate is not None and abs(float(f_rate) - float(c_rate)) < 1e-9)
        if not same_rate or f_n != c_n:
            bad.append(f"{tid}: the fit saw {_clip_txt(f_rate, f_n)}, the current metrics give {_clip_txt(c_rate, c_n)} — the fit predates these metrics")
    if bad:
        return (f"is fit over other rows than the current round-{','.join(map(str, sorted(rounds)))} summaries ({'; '.join(bad)}) — "
                f"re-run the analysis over the current evidence")
    return None


def _menu_fit_binding(menu_fit: dict, design_fit: dict | None, allowed: dict | None, source_record: str | None) -> str | None:
    """Why the derived refit may NOT be used for this table (None = bound). The model/class mirror is shared across
    --cluster and the fetch has no --delete, so a residual r1_fit_menu.json from another cluster or menu generation must never pivot this
    search — it is bound to the design fit (the same search id when both carry one, and the SAME rows: trial ids + y), to the current
    menu (the menu file's sha and the probe's result generation) and to the source record (<cluster>:<record>) it was derived on."""
    d_keys = sorted(map(tuple, (design_fit or {}).get("allowed_pairs") or []))
    m_keys = sorted(map(tuple, menu_fit.get("allowed_pairs") or []))
    m_from = sorted(map(tuple, menu_fit.get("menu_extended_from") or []))
    if menu_fit.get("design") != "v2" or not m_keys or not m_from or m_from != d_keys or not set(d_keys) <= set(m_keys):
        return (f"is not a widened-menu refit of the design fit (design={menu_fit.get('design')!r}, extended_from={m_from}, "
                f"design menu={d_keys}, menu={m_keys})")
    if design_fit is not None:
        d_sid, m_sid = design_fit.get("search_id"), menu_fit.get("search_id")
        if d_sid and m_sid and str(d_sid) != str(m_sid):
            return f"is derived from search {m_sid}, the design fit is search {d_sid}"
        if _fit_identity(design_fit) != _fit_identity(menu_fit):
            return "is fit over other rows than the design fit's (trial ids / held-out values differ) — another search's or another record's trials"
    if allowed is not None:
        ms = menu_fit.get("menu_source") or {}
        if not ms.get("sha256"):
            return "records no menu_source — re-derive it (`analyze --design v2 --allowed-pairs <menu> --extends-menu --fit-name r1_fit_menu.json`)"
        if ms.get("sha256") != allowed.get("sha256"):
            return f"is derived over menu sha {str(ms.get('sha256'))[:16]}, the current menu is {str(allowed.get('sha256'))[:16]}"
        if allowed.get("result_generation") and ms.get("result_generation") and ms["result_generation"] != allowed["result_generation"]:
            return f"is derived over menu generation {str(ms['result_generation'])[:12]}, the current menu is {str(allowed['result_generation'])[:12]}"
    if source_record:
        got = menu_fit.get("source_record")
        if not got:
            return f"records no source_record; this table is {source_record}"
        if str(got) != str(source_record):
            return f"is derived on {got}, this table is {source_record} (a residual refit on the shared mirror)"
    return None


def load_summaries(search_dir: Path) -> list[dict]:
    """Every trial summary under the search dir with its CURRENT clip evidence (read from the trial's own metrics.jsonl beside it) — the view the
    table lists and the view every fit is bound to (hpo_pick reads the same)."""
    summaries: list[dict] = []
    for p in sorted(search_dir.glob("r*_t*/summary.json")):
        s = json.loads(p.read_text())
        s["_trial_id"] = p.parent.name
        s["_dir"] = str(p.parent)
        rate, n_steps = tune_arm._trial_clip_activation(p.parent, s)
        s["_clip_fraction"], s["_clip_n_steps"] = rate, n_steps
        summaries.append(s)
    return summaries


def load_search(search_dir: Path, allowed_path: str | None = None, source_record: str | None = None) -> dict:
    """Everything the table needs, read from the search directory: the
    summaries (every round), the design fit, the pair-only refit, the
    allowed pairs (--allowed-pairs, else the design fit's / summaries' own
    menu), the followup evidence."""
    summaries = load_summaries(search_dir)
    fits = {}
    for name in ("r1_fit.json", "r1+2_fit_pair.json", MENU_FIT_NAME):
        f = search_dir / name
        if f.is_file():
            fits[name] = json.loads(f.read_text())
    for name, fit in fits.items():            # no fit stands over summaries other than the ones it was fit from
        why = _fit_rows_binding(fit, summaries)
        if why:
            raise SystemExit(f"[hpo-table] {name} {why} — refusing to publish a table whose rows and pivot disagree")
    hold_p = search_dir / "hpo_pick.HOLD.json"                 # the pick is HELD — the table says so where a promoted row would be
    pick_hold = json.loads(hold_p.read_text()) if hold_p.is_file() else None
    design_fit = fits.get("r1_fit.json")
    refit = fits.get("r1+2_fit_pair.json")
    # the round-1 rows refit over a WIDENED menu (03 --phase addendum writes it beside the design fit; `analyze --extends-menu
    # --fit-name`). It supplies the pivot's predictions over the widened menu; the DESIGN fit keeps the round-1 selection, its alarms
    # and the provenance the pick was made from. Accepted only when it is that: a v2 artefact over exactly the design's menu plus more.
    menu_fit = fits.get(MENU_FIT_NAME)
    allowed_for_binding = tune_arm.load_allowed_pairs(allowed_path) if allowed_path else None
    if menu_fit is not None:
        why = _menu_fit_binding(menu_fit, design_fit, allowed_for_binding, source_record)
        if why:
            raise SystemExit(f"[hpo-table] {MENU_FIT_NAME} {why} — refusing to pivot over it (move it aside or re-derive it)")
    allowed = None
    if allowed_path:
        allowed = tune_arm.load_allowed_pairs(allowed_path)
    keys = None
    excluded = []
    set_order = None
    if allowed is not None:
        keys = tune_arm.allowed_pair_keys(allowed)
        excluded = [[e["rank"], e["set"], e["reason"]] for e in allowed["excluded"]]
        set_order = tune_arm.set_order_for_pairs(allowed)
    else:
        src = design_fit or {}
        if src.get("allowed_pairs"):
            keys = [tuple(tune_arm.pair_key(a, b)) for a, b in src["allowed_pairs"]]
            excluded = src.get("excluded_pairs") or []
            set_order = src.get("set_order")
        else:
            for s in summaries:
                if s.get("allowed_pairs"):
                    keys = [tuple(tune_arm.pair_key(a, b)) for a, b in s["allowed_pairs"]]
                    excluded = s.get("excluded_pairs") or []
                    set_order = s.get("set_order")
                    break
    if allowed is not None and design_fit and \
            design_fit.get("allowed_pairs") is not None and \
            sorted(map(tuple, design_fit["allowed_pairs"])) != sorted(keys):
        # a WIDENED menu is accepted only through the derived fit that was refit over exactly it
        if menu_fit is None or sorted(map(tuple, menu_fit.get("allowed_pairs") or [])) != sorted(keys):
            raise SystemExit("[hpo-table] --allowed-pairs disagrees with the menu the "
                             "design fit was analysed over — one search, one design "
                             f"space (a widened menu needs {MENU_FIT_NAME}, the refit over exactly that menu: "
                             "`analyze --design v2 --allowed-pairs <menu> --extends-menu --fit-name r1_fit_menu.json`)")
    elif menu_fit is not None and allowed is None:
        keys = [tuple(tune_arm.pair_key(a, b)) for a, b in menu_fit["allowed_pairs"]]      # the widened menu is the current one
        excluded = menu_fit.get("excluded_pairs") or excluded
        set_order = menu_fit.get("set_order") or set_order
    # trainable_params before a trial ran: the probe's count per pair
    pair_params = {}
    if allowed is not None:
        for pr in allowed["pairs"]:
            if pr.get("trainable_params") is not None:
                if pr.get("trainable_params") is not None:                   # An override pair's params are known only once its trial runs
                    pair_params[tune_arm.pair_key(pr["rank"], pr["set"])] = int(pr["trainable_params"])
    # the DESIGN the trials were drawn from (its lr box after the reconnaissance's clamp — the table's header carries the
    # width) and the reconnaissance verdict beside it, when the search dir holds them (a search rendered from summaries alone has neither)
    design = None; recon = None
    for name, slot in (("r1_trials.json", "design"), ("recon.json", "recon")):
        f = search_dir / name
        if f.is_file():
            try:
                doc = json.loads(f.read_text())
            except json.JSONDecodeError:
                doc = None
            if slot == "design": design = doc
            else: recon = doc
    return {"design": design, "recon": recon,
            "summaries": summaries, "design_fit": design_fit, "menu_fit": menu_fit, "refit": refit,
            "fits": fits, "allowed_keys": keys, "excluded": excluded,
            "set_order": set_order, "pair_params": pair_params,
            # the artefact whose guard names the selected trial: the pair-only
            # refit when it ran, else the design fit
            "final": refit or design_fit,
            "search_dir": str(search_dir), "pick_hold": pick_hold}


def _row_of(s: dict, search: dict, final: dict | None) -> dict:
    t = s.get("trial") or {}
    key = tune_arm._pair_of(t)
    role = ("confirmation" if s.get("confirmation_of")
            else s.get("trial_role") or ("refine" if s.get("refine_of") else "search"))
    alarms = []
    if final is not None:
        g = final.get("selection_guard") or {}
        if g.get("selected_trial") == s["_trial_id"]:
            alarms = [guard_mod.alarm_code(a) for a in g.get("alarms") or []]
    if s.get("exploration_replicate_of"):
        ro = s["exploration_replicate_of"]
        y = s.get("best_heldout_loss")
        note = f"exploration_replicate of {ro.get('trial')}"
        if y is not None and ro.get("original_y") is not None and s.get("status") == "ok":
            note += f"; replicate noise {float(y) - float(ro['original_y']):+.5f} CE"
        note += " (NOT a confirmation)"
        alarms.append(note)
    if s.get("confirmation_of") and final is not None:
        fu = final.get("followup") or {}
        c = fu.get("confirmation") or {}
        if c.get("trial_id") == s["_trial_id"] and c.get("within_noise") is not None:
            alarms.append(f"confirmation_within_noise={c['within_noise']}")
    tp = s.get("trainable_params")
    if tp is None and key is not None:
        tp = search["pair_params"].get(key)
    curve = s.get("heldout_curve") or []
    return {
        "trial_id": s["_trial_id"], "round": s.get("round"), "role": role,
        "rank": key[0] if key else None, "set": key[1] if key else None,
        "per_expert_rank": t.get("per_expert_rank"),
        "trainable_params": tp, "lr": t.get("lr"), "warmup": t.get("warmup_frac"),
        "clip": t.get("max_grad_norm"), "best_heldout_CE": s.get("best_heldout_loss"),
        "epochs": (s.get("epochs_seen") if s.get("epochs_seen") is not None
                   else (max(len(curve) - 1, 0) if curve else None)),
        "clip_fraction": s.get("_clip_fraction"), "status": s.get("status"),
        "alarms": "; ".join(alarms),
        "_outside_allowed": (key is not None and search["allowed_keys"] is not None
                             and key not in set(search["allowed_keys"])),
    }


def build_rows(search: dict) -> list[dict]:
    rows = [_row_of(s, search, search["final"]) for s in search["summaries"]]
    bad = [r["trial_id"] for r in rows if r["_outside_allowed"]]
    if bad:
        raise SystemExit(f"[hpo-table] trials {bad} trained a (rank, set) OUTSIDE "
                         "the allowed pairs — a defect, refusing to table them "
                         "as candidates")
    return rows


def build_pivot(search: dict) -> dict:
    """{ranks, sets, cells: {(rank,set): {...}}, model_dependent, winner}
    of the FITTED beta+gamma prediction at u* over the allowed pairs (the
    design fit's pair_predictions — the widened-menu refit's when an
    addendum widened the menu); excluded pairs greyed with the reason."""
    pred_src = search.get("menu_fit") or search["design_fit"] or {}
    fit = pred_src.get("fit") or {}
    preds = fit.get("pair_predictions") or {}
    allowed = set(search["allowed_keys"] or [])
    excluded = {tune_arm.pair_key(e[0], e[1]): e[2] for e in (search["excluded"] or [])}
    every = allowed | set(excluded)
    ranks = sorted({int(k[0]) for k in every})
    sets_seen = {k[1] for k in every}
    order = search["set_order"]
    if order is None and every:
        raise SystemExit("[hpo-table] the search records no module-set order "
                         "(set_order on the design fit / summaries, or "
                         "--allowed-pairs with trainable_params) — the order is "
                         "derived from counts (plan §3.10), never hand-written")
    order = order or []
    sets = [s for s in order if s in sets_seen] + sorted(sets_seen - set(order))
    adequacy = (search["design_fit"] or {}).get("adequacy") or {}
    model_dependent = bool(adequacy.get("alarm"))
    sel = (search["design_fit"] or {}).get("selection") or {}
    winner_key = (tune_arm.pair_key(sel["lora_rank"], sel["lora_set_id"])
                  if sel.get("lora_rank") is not None else None)
    cells: dict = {}
    for r in ranks:
        for sid in sets:
            k = (r, sid)
            if k in excluded:
                cells[k] = {"state": "excluded", "reason": excluded[k]}
            elif k in allowed:
                p = preds.get(f"{r}:{sid}") or {}
                cells[k] = {"state": "allowed", "pred_y": p.get("pred_y"),
                            "u_star": p.get("u_star"), "lr": p.get("lr"),
                            "interior": p.get("interior"), "n_rows": p.get("n_rows"),
                            "winner": (k == winner_key)}
            else:
                cells[k] = {"state": "unprobed"}
    return {"ranks": ranks, "sets": sets, "cells": cells,
            "model_dependent": model_dependent, "winner": winner_key,
            "positive_definite": fit.get("positive_definite"),
            "interaction_used": fit.get("interaction_used"),
            "predictions_from": (MENU_FIT_NAME if search.get("menu_fit") else "r1_fit.json"),
            "menu_extended_from": (search.get("menu_fit") or {}).get("menu_extended_from")}


def alarm_lines(search: dict) -> list[str]:
    out = []
    if search.get("menu_fit"):
        mf = search["menu_fit"]
        out.append(f"[{MENU_FIT_NAME}] the rank x set pivot's predictions come from the round-1 rows REFIT over the widened menu "
                   f"(extension {mf.get('menu_extension')}); the round-1 selection, alarms and provenance below are the design fit's")
    for name in ("r1_fit.json", "r1+2_fit_pair.json"):
        art = search["fits"].get(name)
        if not art:
            continue
        g = art.get("selection_guard") or {}
        out.append(f"[{name}] " + (f"selection: {g.get('selected_from')} "
                                   f"{g.get('selected_trial') or ''}").strip())
        for line in g.get("text") or []:
            out.append(f"[{name}] {line}")
        for line in (art.get("adequacy") or {}).get("text") or []:
            out.append(f"[{name}] {line}")
    if search.get("pick_hold"):
        h = search["pick_hold"]
        out.append(f"PICK HELD (hpo_pick.HOLD.json, {h.get('written_utc')}): {h.get('reason')} — the held pick is rank "
                     f"{(h.get('held_pick') or {}).get('rank')} / set {(h.get('held_pick') or {}).get('set')}; every consumer (04's LoRA fits, "
                     f"the fold, this table's promotion) refuses by name until a fresh --phase pick re-derives a promotion from the current evidence")
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in COLUMNS])


def design_lines(search: dict) -> list[str]:
    """The design's header: the lr box the trials were drawn from with its width in decades (the reconnaissance's certified top
    clamps the seed box; the bottom stays), the clip constant, and the reconnaissance's certified lr top with its stable / unstable rungs."""
    out: list[str] = []
    d = search.get("design") or {}; r = search.get("recon") or {}
    box = (d.get("box") or {})
    lr = box.get("lr")
    if isinstance(lr, (list, tuple)) and len(lr) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in lr):
        lo, hi = float(lr[0]), float(lr[1]); width = math.log10(hi) - math.log10(lo)
        out.append(f"lr box [{lo:.3e}, {hi:.3e}] (log10 [{math.log10(lo):.3f}, {math.log10(hi):.3f}]; {width:.2f} decades)"
                   + ("" if width >= 2.5 else " — below the HPO protocol's advisory 2.5-decade width; the top is the reconnaissance's certified lr, the bottom stays (D157)"))
    clip = box.get("max_grad_norm")
    if isinstance(clip, (list, tuple)) and len(clip) == 2 and clip[0] == clip[1]:
        out.append(f"clip constant {clip[0]!r}" + (f" (the reconnaissance's p{r.get('percentile'):g} of the pooled pre-clip norm)" if r.get("percentile") is not None else ""))
    if r.get("certified_lr_max") is not None:
        out.append(f"reconnaissance certified lr top {float(r['certified_lr_max']):.4e} (log10 {float(r['certified_lr_log10_max']):.3f}); "
                   f"stable rungs {[f'{float(x):.3e}' for x in r.get('stable_lrs') or []]}; unstable {[f'{float(x):.3e}' for x in r.get('unstable_lrs') or []]}")
    if d.get("axes") is not None:
        out.append(f"searched axes {d.get('axes')}; warm-up levels {d.get('warmup_levels')}; init {d.get('init_scheme')}; cell {d.get('si')}; round-1 trials {len(d.get('trials') or [])}")
    return out


def write_md(rows: list[dict], pivot: dict, lines: list[str], path: Path,
             search: dict) -> None:
    md = [f"# HPO table — {search['search_dir']}", ""]
    dl = design_lines(search)
    if dl:
        md.append("## design"); md.append(""); md.extend(f"- {ln}" for ln in dl); md.append("")
    md.append("| " + " | ".join(COLUMNS) + " |")
    md.append("|" + "---|" * len(COLUMNS))
    for r in rows:
        md.append("| " + " | ".join(_fmt(r[c]).replace("|", "/") for c in COLUMNS) + " |")
    md.append("")
    if pivot["ranks"]:
        md.append("## rank x module set — fitted beta+gamma prediction at u* (held-out CE; lower is better)"
                  + (" — MODEL-DEPENDENT (additive_model_inadequate)" if pivot["model_dependent"] else ""))
        md.append("")
        md.append("| rank \\ set | " + " | ".join(pivot["sets"]) + " |")
        md.append("|" + "---|" * (len(pivot["sets"]) + 1))
        for r in pivot["ranks"]:
            cells = []
            for sid in pivot["sets"]:
                c = pivot["cells"][(r, sid)]
                if c["state"] == "excluded":
                    cells.append(f"excluded: {c['reason']}")
                elif c["state"] == "unprobed":
                    cells.append("not probed")
                else:
                    v = _fmt(c.get("pred_y"), ".5f") or "n/a"
                    cells.append(f"{v}{' (winner)' if c.get('winner') else ''}"
                                 f"{' *' if pivot['model_dependent'] else ''}")
            md.append(f"| {r} | " + " | ".join(cells) + " |")
        md.append("")
    md.append("## alarms and adequacy (verbatim)")
    md.append("")
    md.extend(f"- {ln}" for ln in lines)
    md.append("")
    path.write_text("\n".join(md))


_CSS = """
:root { --ink: #1f2328; --ink-2: #57606a; --ink-3: #8c959f; --line: #d0d7de;
        --surface: #ffffff; --surface-2: #f6f8fa; --winner: #0a3d62; --alarm: #9a3412; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #e6edf3; --ink-2: #9ea7b3; --ink-3: #6e7781; --line: #30363d;
          --surface: #0d1117; --surface-2: #161b22; --winner: #9ecbff; --alarm: #ffa657; } }
body { font: 14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: var(--ink);
       background: var(--surface); margin: 2rem auto; max-width: 1280px; padding: 0 1rem; }
h1, h2 { font-weight: 600; } h1 { font-size: 1.35rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
caption { text-align: left; color: var(--ink-2); padding: .4rem 0; }
th, td { border-bottom: 1px solid var(--line); padding: .35rem .5rem; text-align: right; }
th { color: var(--ink-2); font-weight: 600; background: var(--surface-2); position: sticky; top: 0; }
td.text, th.text { text-align: left; } td.num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
tr.winner td { font-weight: 600; color: var(--winner); }
td.excluded { color: var(--ink-3); background: var(--surface-2); font-style: italic; }
td.unprobed { color: var(--ink-3); }
.alarm { color: var(--alarm); }
.note { color: var(--ink-2); }
ul.alarms li { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
"""


def write_html(rows: list[dict], pivot: dict, lines: list[str], path: Path,
               search: dict) -> None:
    e = html.escape
    h = ["<!doctype html>", "<html lang=\"en\"><head><meta charset=\"utf-8\">",
         f"<title>HPO table — {e(Path(search['search_dir']).name)}</title>",
         f"<style>{_CSS}</style></head><body>",
         f"<h1>LoRA architecture search — {e(Path(search['search_dir']).name)}</h1>",
         "<p class=\"note\">One row per trial. best_heldout_CE is the trial's best "
         "per-epoch held-out cross-entropy (lower is better); clip_fraction the share "
         "of its optimizer steps whose pre-clip gradient norm exceeded its clip; "
         "trainable_params the built object's trainable scalars. An exploration "
         "replicate is replicate-noise evidence, never a confirmation.</p>",
         *([f"<h2>design</h2><ul class=\"design\">" + "".join(f"<li>{e(ln)}</li>" for ln in design_lines(search)) + "</ul>"] if design_lines(search) else []),
         "<table><caption>Trials</caption><thead><tr>"]
    text_cols = {"trial_id", "role", "set", "per_expert_rank", "status", "alarms"}
    for c in COLUMNS:
        h.append(f"<th scope=\"col\" class=\"{'text' if c in text_cols else ''}\">{e(c)}</th>")
    h.append("</tr></thead><tbody>")
    sel_tid = ((search["final"] or {}).get("selection_guard") or {}).get("selected_trial")
    for r in rows:
        cls = " class=\"winner\"" if r["trial_id"] == sel_tid else ""
        h.append(f"<tr{cls}>")
        for c in COLUMNS:
            v = _fmt(r[c], ".5g" if c in ("lr", "best_heldout_CE") else ".4g")
            klass = "text" if c in text_cols else "num"
            if c == "alarms" and v:
                klass += " alarm"
            h.append(f"<td class=\"{klass}\">{e(v)}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    if pivot["ranks"]:
        dep = (" — <span class=\"alarm\">MODEL-DEPENDENT: additive_model_inadequate</span>"
               if pivot["model_dependent"] else "")
        h.append(f"<h2>rank × module set: fitted β+γ prediction at u* (held-out CE, lower is better){dep}</h2>")
        h.append("<p class=\"note\">Allowed pairs only. An excluded pair shows the serving "
                 "probe's reason and is not a candidate. Set order: narrowest → widest by "
                 "trainable-parameter count.</p>")
        h.append("<table><caption>Fitted pair predictions</caption><thead><tr>"
                 "<th scope=\"col\" class=\"text\">rank \\ set</th>")
        for sid in pivot["sets"]:
            h.append(f"<th scope=\"col\">{e(sid)}</th>")
        h.append("</tr></thead><tbody>")
        for rk in pivot["ranks"]:
            h.append(f"<tr><th scope=\"row\" class=\"text\">{rk}</th>")
            for sid in pivot["sets"]:
                c = pivot["cells"][(rk, sid)]
                if c["state"] == "excluded":
                    h.append(f"<td class=\"excluded\">excluded: {e(c['reason'])}</td>")
                elif c["state"] == "unprobed":
                    h.append("<td class=\"unprobed\">not probed</td>")
                else:
                    v = _fmt(c.get("pred_y"), ".5f") or "n/a"
                    tag = " (winner)" if c.get("winner") else ""
                    lr = f" @ lr {_fmt(c.get('lr'), '.3g')}" if c.get("lr") is not None else ""
                    h.append(f"<td class=\"num{' winner' if c.get('winner') else ''}\">"
                             f"{e(v)}{e(tag)}<br><span class=\"note\">{e(lr.strip())}</span></td>")
            h.append("</tr>")
        h.append("</tbody></table>")
    h.append("<h2>Alarms and model adequacy (verbatim)</h2><ul class=\"alarms\">")
    for ln in lines:
        klass = " class=\"alarm\"" if "ALARM" in ln else ""
        h.append(f"<li{klass}>{e(ln)}</li>")
    h.append("</ul></body></html>")
    path.write_text("\n".join(h))


def render(search_dir: Path, out_dir: Path, allowed_path: str | None = None, source_record: str | None = None) -> dict:
    search = load_search(search_dir, allowed_path, source_record)
    rows = build_rows(search)
    pivot = build_pivot(search)
    lines = alarm_lines(search)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "table.csv")
    write_md(rows, pivot, lines, out_dir / "table.md", search)
    write_html(rows, pivot, lines, out_dir / "table.html", search)
    (out_dir / "table.json").write_text(json.dumps({
        "design": design_lines(search),
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "pivot": {"ranks": pivot["ranks"], "sets": pivot["sets"],
                  "cells": {f"{k[0]}:{k[1]}": v for k, v in pivot["cells"].items()},
                  "model_dependent": pivot["model_dependent"],
                  "winner": list(pivot["winner"]) if pivot["winner"] else None},
        "alarms": lines, "search_dir": str(search_dir)}, indent=2, default=str) + "\n")
    return {"rows": rows, "pivot": pivot, "alarms": lines}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--search-dir", required=True)
    ap.add_argument("--allowed-pairs", default=None)
    ap.add_argument("--source-record", default=None, help="<cluster>:<record dir> of the trials this table renders; a derived refit bound to another is refused")
    ap.add_argument("--out", default=None, help="the table directory (required unless --verify-only)")
    ap.add_argument("--verify-only", action="store_true",
                    help="load the search and exit 0 when every fit stands over the current evidence (the bindings refuse by name otherwise); renders nothing")
    a = ap.parse_args(argv)
    if a.verify_only:
        search = load_search(Path(a.search_dir), a.allowed_pairs, a.source_record)
        n_fits = sum(1 for k in ("design_fit", "refit", "menu_fit") if search.get(k))
        print(f"[hpo-table] every fit ({n_fits}) stands over the current evidence ({len(search['summaries'])} trials)"
              + (" — the pick is HELD (hpo_pick.HOLD.json)" if search.get("pick_hold") else ""))
        return 0
    if not a.out:
        ap.error("--out is required unless --verify-only")
    res = render(Path(a.search_dir), Path(a.out), a.allowed_pairs, a.source_record)
    print(f"[hpo-table] {len(res['rows'])} trial rows, "
          f"{len(res['pivot']['ranks'])} x {len(res['pivot']['sets'])} pivot, "
          f"{len(res['alarms'])} alarm/adequacy lines -> {a.out}/table.{{csv,md,html}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
