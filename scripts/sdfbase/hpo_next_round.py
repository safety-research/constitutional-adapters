#!/usr/bin/env python
"""hpo_next_round — what follows ONE analyze artefact: RECENTER, CONFIRM or
STOP.

    python scripts/sdfbase/hpo_next_round.py \\
        --fit $STAGE/sdfbase/llama8b/hpo_out/llama8b_base/psr_ce_all_all/r1_fit.json \\
        --config configs/sdfbase_llama8b_base_fits.yaml --model-key llama8b_base \\
        --arm psr_ce_all --out-dir $STAGE/sdfbase/hpo/trials

The edge policy, as code, read off ``tune_arm.py analyze``'s
``r<tag>_fit.json``:

* **recenter** — the quadratic is positive-definite but its argmin sits on
  the box edge, and only ONE search round has been pooled: emit exactly one
  recenter round (``emit-trials --box-from``, the artefact's ``next.box`` =
  same log-width re-centred on the argmin) with the standing trial count —
  8 for the LR-only design (its round IS 8), 16 for an lr x clip design.
  The refine round fires only on a box edge.
* **confirm** — the argmin is interior (``winner`` populated): emit the
  ONE-row confirmation design at exactly that point (``emit-trials
  --confirm-from``, round 9). The fitted argmin is a point no trial trained
  at; the spec requires the promoted point to be one that actually ran, so
  the confirmation is what makes it promotable — never a fresh search.
* **stop** — nothing further is emitted: the fit is rank-deficient,
  ill-conditioned or not positive-definite (no fitted minimum: the spec's
  conservative fall-through to the best trial that RAN, inside the box and
  below the clip-activation threshold, applies at fold time); the argmin is
  STILL on the edge after two pooled rounds (no third round, ever); or the
  interior winner's selection guard is ALARMED (an unacknowledged alarm
  cannot fold — acknowledge deliberately with ``--allow-alarmed`` here,
  which records the acknowledgement requirement and proceeds to confirm).

**The argmin guard is called, never bypassed.** Before deciding anything
the driver re-runs ``hpo_selection_guard.guard_selection`` on the artefact's
own rows / box / fit for the point the artefact says it filed, re-derives
the analyze-level clip-activation alarm from the rows' own rates, and
refuses if the recomputed alarms differ from the recorded ones OR the
recorded ``alarmed`` flag disagrees with the recomputed set (a hand-edited
or stale artefact cannot route a round; the decision honours the RECOMPUTED
verdict, never a stored boolean). It also checks that the three copies of
the selected point agree — ``fit.argmin`` (what was fitted), ``selection``
(what the guard checked) and ``winner`` (what a confirmation would be
EMITTED at) — so the driver can never authorize paid work at a point the
guard did not check.

Prints the decision and its evidence (rounds pooled, usable / excluded
trials, the fit's health, the edge positions, the guard's alarms and their
reproduction, the winner and its clip-activation neighbourhood) as text and
JSON, writes the decision record beside the artefact
(``r<tag>_next_round.json``) and the emitted design into ``--out-dir`` as
``r<round>_<model-key>_<arm>_all.json`` (the name ``hpo-spawn --round``
reads). Exit 0 on recenter / confirm, EXIT_STOP (3) on stop, 1 on a
refusal. ``--dry-run`` decides without emitting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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


# tune_arm owns emit-trials, the trial-count rule and the confirmation
# design; the guard rides on it (tune_arm.hpo_selection_guard) so the driver
# and analyze run ONE guard module instance.
tune_arm = (sys.modules.get("_tune_arm") or sys.modules.get("tune_arm")
            or _load_by_path("tune_arm", SCRIPTS / "tune_arm.py"))
guard_mod = tune_arm.hpo_selection_guard

#: The recenter round (exactly one, after round 1).
RECENTER_ROUND = 2
#: The confirmation round (reserved; tune_arm.CONFIRMATION_ROUND).
CONFIRM_ROUND = tune_arm.CONFIRMATION_ROUND
#: Exit code for a `stop` decision (a decision, not an error — but a shell
#: chain that would `volume put` a design must not continue).
EXIT_STOP = 3

DECISIONS = ("recenter", "confirm", "stop")


def _codes(alarms) -> list[str]:
    """Sorted alarm codes of an alarm list, rendered by the guard's ONE
    ``alarm_code`` (so the recorded and recomputed sets are spelled alike)."""
    return sorted({guard_mod.alarm_code(a) for a in (alarms or [])})


POINT_KEYS = ("lr", "max_grad_norm", "warmup_frac")
EXTRA_AXES = ("temperature", "weight_decay", "lora_rank")


def _same_point(a: dict, b: dict, keys=POINT_KEYS) -> bool:
    """The two point records agree on every coordinate in ``keys`` (to
    1e-12 relative; analyze copies them verbatim, so equality is exact
    through JSON — the tolerance only guards against a re-serialisation)."""
    try:
        return all(abs(float(a[k]) - float(b[k]))
                   <= 1e-12 * max(1.0, abs(float(b[k]))) for k in keys)
    except (KeyError, TypeError, ValueError):
        return False


def _fmt_point(d: dict, keys=POINT_KEYS) -> str:
    return " ".join(f"{k}={d.get(k)!r}" for k in keys)


def reproduce_guard(art: dict) -> dict:
    """Re-run the argmin guard on the artefact's own evidence for the point
    its record says it filed (the fitted argmin, or the named raw trial),
    re-derive the clip-activation alarm from the rows' own rates, and
    compare: {reproduced, recorded_codes, recomputed_codes, alarmed
    (RECOMPUTED), recorded_alarmed, selected_from, selected_trial}.

    Refuses (SystemExit) when the artefact has no guard record; when the
    recomputed alarm set, the checked point or the alarmed flag differs
    from the record; when a fitted-argmin selection is not the fit's own
    argmin; or when the ``winner`` block — the point a confirmation is
    EMITTED at — is not the guard-checked selection. Every copy of the
    point a later step reads is compared here, so no downstream read can
    authorize an unchecked point."""
    g = art.get("selection_guard")
    if not isinstance(g, dict):
        raise SystemExit("[hpo-next-round] the artefact carries no "
                         "selection_guard record — re-run tune_arm.py "
                         "analyze on this tree (the guard is not optional)")
    fit = art["fit"]
    rows = art["rows"]
    box = art["box"]
    levels = tuple(fit["levels"])
    y_key = art["y_key"]
    baseline = (g.get("spread") or {}).get("baseline")
    common = dict(rows=rows, box=box, levels=levels, fit=fit, y_key=y_key,
                  sigma_log_y=fit.get("sigma_log_y"), baseline=baseline)
    # A `--design v2` artefact (the LoRA architecture search) records the NAMED
    # discrete menus its guard judged against (`selection_guard.discrete_axes`:
    # lora_rank / lora_modules with levels + the selected level); passing the
    # SAME axes back is what makes the recomputed alarm set reproduce the
    # record (discrete_extreme:<axis>). A v1 artefact carries none and is
    # judged exactly as before (the key is absent, nothing is passed).
    if isinstance(g.get("discrete_axes"), dict) and g["discrete_axes"]:
        common["discrete_axes"] = g["discrete_axes"]
    sel = art.get("selection") or {}
    if g.get("selected_from") == "fitted_argmin":
        # The checked point must BE the fitted argmin: the selection block
        # is analyze's verbatim copy of fit.argmin, so a disagreement is an
        # edit to one copy or the other.
        argmin = fit.get("argmin") or {}
        if not _same_point(sel, argmin):
            raise SystemExit(
                "[hpo-next-round] the artefact's selection block does NOT "
                f"reproduce its fitted argmin (selection {_fmt_point(sel)}; "
                f"fit.argmin {_fmt_point(argmin)}) — a stale or hand-edited "
                "artefact cannot route a round; re-run analyze")
        rg = guard_mod.guard_selection(
            selected=sel, selected_from="fitted_argmin",
            fitted_min_log_y=sel.get("fitted_min_log_y"), **common)
    else:
        tid = g.get("selected_trial")
        row = next((r for r in rows if guard_mod.trial_id(r) == tid), None)
        if row is None:
            raise SystemExit(f"[hpo-next-round] the guard names raw trial "
                             f"{tid!r}, which is not among the artefact's "
                             "usable rows")
        rg = guard_mod.guard_selection(
            selected=row, selected_from="best_raw_trial",
            fall_through=g.get("fall_through"), **common)
    # The analyze-level clip-activation alarm (tune_arm._stamp_clip_activation
    # files it OUTSIDE guard_selection): re-derived here from the rows' own
    # per-trial rates — the rows are the artefact's evidence, like y — so
    # the FULL alarm set is recomputed and nothing is taken from a flag.
    clip_act = tune_arm.clip_activation_summary(rows, y_key=y_key)
    alarms = list(rg["alarms"])
    if clip_act.get("all_known_over_threshold"):
        alarms.append({"code": tune_arm.CLIP_ACTIVATION_ALARM, "dim": None})
    recorded, recomputed = _codes(g.get("alarms")), _codes(alarms)
    same_point = _same_point(rg["selected"], g.get("selected") or {})
    alarmed = bool(alarms)
    recorded_alarmed = bool(g.get("alarmed"))
    out = {"reproduced": (recorded == recomputed and same_point
                          and alarmed == recorded_alarmed),
           "recorded_codes": recorded, "recomputed_codes": recomputed,
           # kept for readers of earlier decision records: the recorded
           # set is now the FULL set (the clip alarm is reproduced too)
           "recorded_codes_all": recorded,
           "alarmed": alarmed,               # RECOMPUTED, never the stored flag
           "recorded_alarmed": recorded_alarmed,
           "selected_from": g.get("selected_from"),
           "selected_trial": g.get("selected_trial"),
           "acknowledged": g.get("acknowledged")}
    if not out["reproduced"]:
        raise SystemExit(
            "[hpo-next-round] the artefact's selection-guard record does NOT "
            f"reproduce from its own rows (recorded {recorded}, recomputed "
            f"{recomputed}, same point {same_point}, recorded alarmed "
            f"{recorded_alarmed} vs recomputed {alarmed}) — a stale or "
            "hand-edited artefact cannot route a round; re-run analyze")
    # The point a confirmation would be EMITTED at is the winner block,
    # which emit-trials --confirm-from reads: it must be the checked point
    # on every coordinate, the echoed extra axes included.
    win = art.get("winner")
    if win is not None:
        if g.get("selected_from") != "fitted_argmin":
            raise SystemExit(
                "[hpo-next-round] the artefact carries a winner block but its "
                f"guard filed a {g.get('selected_from')!r} selection — a "
                "winner exists only for an interior fitted argmin; re-run "
                "analyze")
        extra = tuple(ax for ax in EXTRA_AXES if ax in sel)
        if not (_same_point(win, sel) and all(win.get(ax) == sel[ax]
                                              for ax in extra)):
            raise SystemExit(
                "[hpo-next-round] the artefact's winner block does NOT "
                "reproduce the guard-checked selection (winner "
                f"{_fmt_point(win, POINT_KEYS + extra)}; selection "
                f"{_fmt_point(sel, POINT_KEYS + extra)}) — the winner is the "
                "point a confirmation is EMITTED at; refusing to authorize "
                "an unchecked point; re-run analyze")
    return out


def recenter_n(art: dict) -> dict:
    """The standing trial count for the recenter round: 8 for the LR-only
    design, 16 for an lr x clip design — build_trials needs n/2 a
    power of two, which analyze's `next.n` (max(rows//rounds//8*8, floor))
    is not always; the artefact's number is recorded beside the one used."""
    lr_only = tune_arm.lr_only_box(art["box"])
    n = tune_arm.LLAMA8B_CE_TRIALS_PER_ARM if lr_only else 16
    nxt = art.get("next") or {}
    return {"n": n, "lr_only_design": lr_only,
            "artefact_next_n": nxt.get("n"), "artefact_next_action": nxt.get("action")}


def decide(art: dict, *, allow_alarmed: bool = False) -> dict:
    """The decision {decision, reason, evidence} for one artefact (pure:
    reads the artefact, calls the guard, emits nothing)."""
    fit = art.get("fit") or {}
    rounds = [int(r) for r in (art.get("rounds") or [])]
    evidence: dict = {
        "rounds_pooled": rounds,
        "n_usable": len(art.get("rows") or []),
        "n_excluded": art.get("n_excluded"),
        "excluded": art.get("excluded"),
        "y_key": art.get("y_key"),
        "box": art.get("box"),
        "fixed_axes": art.get("fixed_axes"),
        "fit": {k: fit.get(k) for k in (
            "names", "n_rows", "design_rank", "design_cond", "rank_deficient",
            "positive_definite", "hessian_eigs", "sigma_log_y", "argmin",
            "fitted_min_log_y")},
        "selection": art.get("selection"),
        "best_raw_trial": art.get("best_raw_trial"),
        "clip_activation": art.get("clip_activation"),
    }
    if fit.get("rank_deficient"):
        return {"decision": "stop", "reason": "fit_rank_deficient",
                "evidence": evidence,
                "text": "the design matrix is rank-deficient — no fit to act "
                        "on; investigate the exclusions"}
    if float(fit.get("design_cond") or 0.0) > tune_arm.MAX_DESIGN_COND:
        return {"decision": "stop", "reason": "fit_ill_conditioned",
                "evidence": evidence,
                "text": "near-collinear design — the fit cannot be trusted"}
    guard_check = reproduce_guard(art)
    evidence["guard"] = guard_check
    evidence["guard_text"] = (art.get("selection_guard") or {}).get("text")
    if not fit.get("positive_definite"):
        return {"decision": "stop", "reason": "fit_not_positive_definite",
                "evidence": evidence,
                "text": ("the pooled quadratic has no interior minimum (eigs "
                         f"{fit.get('hessian_eigs')}) — no round to emit; the "
                         "spec's conservative fall-through applies at fold "
                         "time: the best trial that RAN, inside the box, "
                         "clip activation below the threshold "
                         f"({(art.get('clip_activation') or {}).get('best_raw_promotable')})")}
    sel = art.get("selection") or {}
    if sel.get("on_edge"):
        if len(rounds) == 1:
            rn = recenter_n(art)
            evidence["recenter"] = rn
            return {"decision": "recenter", "reason": "fitted_argmin_on_edge",
                    "evidence": evidence,
                    "text": (f"fitted argmin on the box edge (edge_pos "
                             f"{sel.get('edge_pos')}) after one round — ONE "
                             f"recenter round of {rn['n']} at the same "
                             "log-width (analyze's next.box)")}
        return {"decision": "stop", "reason": "still_on_edge_after_pooling",
                "evidence": evidence,
                "text": (f"fitted argmin still on the box edge after pooling "
                         f"rounds {rounds} — no third round; the conservative "
                         "fall-through applies at fold time")}
    win = art.get("winner")
    if win is None:
        raise SystemExit("[hpo-next-round] interior selection but no winner "
                         "block — not an analyze artefact this driver "
                         "understands")
    evidence["winner"] = win
    if guard_check["alarmed"] and not allow_alarmed:
        return {"decision": "stop", "reason": "alarmed_interior_winner",
                "evidence": evidence,
                "text": ("the fitted argmin is interior but its selection "
                         f"guard is ALARMED ({guard_check['recorded_codes_all']}) "
                         "— the fold would refuse it unacknowledged; re-run "
                         "with --allow-alarmed to confirm it deliberately "
                         "(the alarms stay on record)")}
    ca = win.get("clip_activation") or {}
    return {"decision": "confirm", "reason": "interior_fitted_argmin",
            "evidence": evidence,
            "text": (f"fitted argmin interior (edge_pos {sel.get('edge_pos')}"
                     f"; fit sigma_log_y {fit.get('sigma_log_y')}) — ONE "
                     f"confirmation trial at exactly lr={win['lr']!r} "
                     f"clip={win['max_grad_norm']!r} "
                     f"warmup={win['warmup_frac']!r}"
                     + (" — clip activation FLAGGED on a nearest trial: the "
                        "confirmation must measure below the threshold"
                        if ca.get("flagged") else "")
                     + (" (alarmed guard acknowledged by --allow-alarmed)"
                        if guard_check["alarmed"] else ""))}


def _design_name(round_no: int, model_key: str, arm: str) -> str:
    return f"r{round_no}_{model_key}_{arm}_all.json"


def emit_for(decision: dict, *, fit_path: Path, config: str, model_key: str,
             arm: str, si: str | None, out_dir: Path,
             recenter_round: int = RECENTER_ROUND,
             confirm_round: int = CONFIRM_ROUND) -> Path | None:
    """Emit the design the decision calls for through tune_arm.emit_trials
    (ONE code path with the CLI); None for stop."""
    if decision["decision"] == "stop":
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    position = str(((decision["evidence"].get("selection") or {}).get("position"))
                   or "all")
    if decision["decision"] == "recenter":
        out = out_dir / _design_name(recenter_round, model_key, arm)
        tune_arm.emit_trials(tune_arm.emit_args(
            config=config, arm=arm, position=position,
            box_from=str(fit_path), round=recenter_round,
            n=decision["evidence"]["recenter"]["n"], si=si, out=str(out)))
        return out
    out = out_dir / _design_name(confirm_round, model_key, arm)
    tune_arm.emit_trials(tune_arm.emit_args(
        config=config, arm=arm, position=position,
        confirm_from=str(fit_path), round=confirm_round, si=si,
        out=str(out)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fit", required=True,
                    help="tune_arm.py analyze's r<tag>_fit.json")
    ap.add_argument("--config", required=True,
                    help="the per-model fits config (emit-trials --config)")
    ap.add_argument("--model-key", required=True,
                    help="the Modal/bank model key (e.g. llama8b_base) — "
                         "names the emitted design r<round>_<key>_<arm>_all.json")
    ap.add_argument("--arm", required=True, choices=["psr_ce_all", "lora_r4_ce"])
    ap.add_argument("--si", default=None,
                    help="the reference data cell; default: the artefact's")
    ap.add_argument("--out-dir", required=True,
                    help="where the emitted design lands (the local mirror "
                         "of the volume's hpo/trials/)")
    ap.add_argument("--recenter-round", type=int, default=RECENTER_ROUND)
    ap.add_argument("--confirm-round", type=int, default=CONFIRM_ROUND)
    ap.add_argument("--allow-alarmed", action="store_true",
                    help="confirm an interior winner whose guard is alarmed "
                         "(the fold still needs the acknowledgement)")
    ap.add_argument("--decision-out", default=None,
                    help="decision record (default: <fit dir>/"
                         "r<tag>_next_round.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and print; emit nothing")
    args = ap.parse_args(argv)

    fit_path = Path(args.fit).expanduser()
    if not fit_path.is_file():
        raise SystemExit(f"[hpo-next-round] no such artefact {fit_path}")
    art = json.loads(fit_path.read_text())
    art_arm = tune_arm.ARM_CANON.get(art.get("arm"), art.get("arm"))
    if art_arm is not None and art_arm != args.arm:
        raise SystemExit(f"[hpo-next-round] the artefact is for arm "
                         f"{art.get('arm')!r}, not --arm {args.arm!r}")
    si = args.si or art.get("si")
    if args.si and art.get("si") and args.si != art["si"]:
        raise SystemExit(f"[hpo-next-round] --si {args.si!r} contradicts the "
                         f"artefact's data cell {art['si']!r}")

    decision = decide(art, allow_alarmed=args.allow_alarmed)
    emitted = None
    if not args.dry_run:
        emitted = emit_for(decision, fit_path=fit_path, config=args.config,
                           model_key=args.model_key, arm=args.arm, si=si,
                           out_dir=Path(args.out_dir).expanduser(),
                           recenter_round=args.recenter_round,
                           confirm_round=args.confirm_round)
    record = {
        "fit": str(fit_path),
        "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        "arm": args.arm, "model_key": args.model_key, "si": si,
        "decision": decision["decision"], "reason": decision["reason"],
        "text": decision["text"],
        "emitted": str(emitted) if emitted else None,
        "emitted_round": (args.recenter_round if decision["decision"] == "recenter"
                          else args.confirm_round if decision["decision"] == "confirm"
                          else None),
        "dry_run": bool(args.dry_run),
        "allow_alarmed": bool(args.allow_alarmed),
        "evidence": decision["evidence"],
    }
    tag = fit_path.stem[:-len("_fit")] if fit_path.stem.endswith("_fit") else fit_path.stem
    dec_path = (Path(args.decision_out).expanduser() if args.decision_out
                else fit_path.parent / f"{tag}_next_round.json")
    dec_path.write_text(json.dumps(record, indent=2) + "\n")

    ev = decision["evidence"]
    print(f"[hpo-next-round] {args.model_key}/{args.arm} rounds "
          f"{ev['rounds_pooled']}: {ev['n_usable']} usable, "
          f"{ev['n_excluded']} excluded; fit PD="
          f"{ev['fit'].get('positive_definite')} eigs="
          f"{ev['fit'].get('hessian_eigs')} sigma_log_y="
          f"{ev['fit'].get('sigma_log_y')}")
    if ev.get("selection"):
        s = ev["selection"]
        print(f"[hpo-next-round] fitted argmin lr={s.get('lr')} "
              f"clip={s.get('max_grad_norm')} warmup={s.get('warmup_frac')} "
              f"on_edge={s.get('on_edge')} edge_pos={s.get('edge_pos')}")
    if ev.get("guard"):
        g = ev["guard"]
        print(f"[hpo-next-round] guard: alarmed={g['alarmed']} "
              f"codes={g['recorded_codes_all']} reproduced={g['reproduced']} "
              f"(selected_from {g['selected_from']})")
        for line in ev.get("guard_text") or []:
            print(f"    {line}")
    print(f"[hpo-next-round] DECISION {decision['decision'].upper()} "
          f"({decision['reason']}): {decision['text']}")
    if emitted:
        print(f"[hpo-next-round] emitted {emitted}")
    print(f"[hpo-next-round] decision record -> {dec_path}")
    print(json.dumps({k: record[k] for k in ("decision", "reason", "emitted",
                                             "emitted_round")}, indent=1))
    return EXIT_STOP if decision["decision"] == "stop" else 0


if __name__ == "__main__":
    sys.exit(main())
