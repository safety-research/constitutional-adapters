#!/usr/bin/env python
"""LOW-DOSE NUMERIC LIVENESS: establish that a served steering object is LIVE by MEASUREMENT rather than by greedy-text
divergence, and record the text probe as an OBSERVATION.

Why this exists. The wave instrument proves a served object live with `steering_bites`: greedy TEXT on five neutral probe
prompts against the path-matched scale-0 twin, refusing when NONE of them differs (`fidelity._check_steering_bites`). At a
small enough dose a CORRECT object changes no greedy token on a neutral prompt — argmax is a step function and a small
logit perturbation crosses no boundary — so that test reports "not live" when steering is live and merely SMALL. The stop
rule takes a failed positive control as a STOP, so as written it would stop the low end of every ladder for being low, and
the low end is where the usable window is expected to be. This is a LOW-DOSE problem, not an AlphaSteer one (the LoRA lane's
realised dose at nominal 0.1 is 0.031 against AlphaSteer's 0.016 at the same nominal, and its 0.1 screens passed).

The rule, as implemented here:
  * LIVE requires TWO numeric facts, both stronger evidence than a token comparison:
      (A) `intervention_hash_distinct` — the steered route's intervention hash differs from the anchor's AND the steered
          responses carry the route's hash (a silently unchanged intervention has been caught on this project by exactly
          this check);
      (B) `continuous_effect` — a MEASURED non-zero effect at the served site, above the engine's own repeat noise. The
          rule is the INSTRUMENT'S OWN: `fidelity.check_logprob_steering_bites`, imported from the wave checkout, three
          legs (steered / scale-0 twin / the twin AGAIN as the noise leg), verdict = steered divergence clears
          max(abs_floor 1e-3, 2 x the measured repeat-noise ceiling). The wave already runs this rule — on MMLU letters,
          inside the controls driver — but the SCREEN never reaches it, because MMLU is not among the screen's controls.
          Here it reads the first generated token's top-K logprobs over the COMMON support of the three legs on the same
          probe prompts, which is that rule applied to the prompts the screen actually has.
  * A cell passing both while FAILING the text probe is LIVE AND SMALL: verdict `live_numeric`, the text result recorded
    as an observation, and the screen proceeds (the wave's refusal branches consult this record — see the policy file
    below). A cell failing either numeric check is genuinely not live: verdict `not_live`, and the screen stops as before.

THE REALISED DOSE IS AN OFFLINE MEASUREMENT, CONSUMED HERE AND LABELLED AS SUCH. Fact (B) is the realised dose
(||edit|| / ||h||, non-zero and scaling with the nominal dose to floating point).
It is NOT measurable at serving time on this code sha: that needs the lens rho record (`SISTEER_LENS_RHO_EMIT` /
`sisteer_rho_record`), which the AlphaSteer campaign's LADDER tree carries and the LoRA screens' wave sha f38d055d does
NOT (verified: no such symbol under src/ or scripts/ at that sha), and serving the screen on a stack that does is exactly
what the identical-instrument rule forbids. But the real quantity WAS MEASURED OFFLINE, on a different path:
`$SHARED/exp/strength_dose/<target>/dose_<target>.json` (`schema strength_dose/1`) carries `realised_dose_median` per
rung, computed analytically from the object bytes against activations captured on the shared reference pool under the
served rendering, with the signed ratio gate per rung. So this tool READS that artifact and, when it BINDS to the
object actually being served, takes fact (B) from it — recorded as `source: "offline_capture"`, never as a serving-time
attestation, because a later reader must not think the serving stack attested it.

The binding is by OBJECT SHA, not by name: the artifact rung's `object_sha256` must equal the sha of the `object.pt` this
job serves. That is what makes the RE-BAKED ladder safe — re-baked objects are different bytes with different layer sets,
so their shas will not match these rungs and the gate refuses to borrow the as-fit numbers, saying the re-bake needs its
own dose capture (that capture is a PREREQUISITE of the re-baked screens, not an optional extra).

Two signals, kept side by side because two independent ones beating either: (B) the offline realised dose (authoritative
for magnitude and scaling) and (C) the serving-time effect above noise (authoritative for "the edit reaches the served
site NOW, on this standup"). (C) is measured and recorded always; it SATISFIES fact (B) only when the artifact is
unavailable — labelled a proxy, because a logprob divergence is not linear in the dose the way ||edit|| is. When both are
evaluable and DISAGREE (offline says the object has magnitude, serving-time sees nothing above noise), the verdict
follows the offline fact — it stands — and the disagreement is flagged loudly in the record and the job log, because
that shape is what a serving bug looks like and it is the thing to investigate if downstream numbers read unsteered.
`--ratio-against <another rung's record>` additionally reports this rung's serving-time effect over another rung's beside
the nominal ratio; reported, never a gate.

Usage (in-job, after the standup, before the positive control; the wrapper does this):
  python scripts/alphasteer_f22/screen/asscr_liveness.py --wave-repo <the CODE checkout> --proxy-url <.../v1> \
      --routing-table <serving dir>/controls_512/routing_table.json --cell alphasteer --tag <TAG> --nominal-dose 0.3 \
      --out <serving dir>/liveness_low_dose.json [--anchor none] [--top-logprobs 64] [--ratio-against <record.json>]
Exit 0 = `live_numeric` (the screen may proceed even if the text probe refuses), 2 = `not_live` (a genuine refusal),
3 = `unevaluable` (the instrument could not be read — never silently a pass).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

VERDICTS = ("live_numeric", "not_live", "unevaluable")
EXIT = {"live_numeric": 0, "not_live": 2, "unevaluable": 3}
#: the record a relaxed refusal branch consults (the wrapper exports it): the wave's smoke and controls driver treat a
#: text-divergence failure as an OBSERVATION only when this names a `live_numeric` record for the same cell and the same
#: routing table — bound to THIS standup, never "the newest record" (the lane's standing lesson).
POLICY_ENV = "SISTEER_LOW_DOSE_LIVENESS_RECORD"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def summary_line(rec: Dict[str, Any]) -> str:
    """The one line the wrapper logs beside the verdict: the three facts, the text observation, and any disagreement."""
    ce = (rec.get("continuous_effect") or {}).get("diag") or {}
    rd = rec.get("realised_dose") or {}
    tx = rec.get("text_divergence_observation") or {}
    dose = (f"offline realised dose {rd.get('realised_dose_median'):.4g} (ratio gate {rd.get('artifact_ratio_gate')}, sha-bound)"
            if rd.get("available") else f"offline realised dose UNAVAILABLE ({str(rd.get('why'))[:80]})")
    eff = (f"serving-time effect median {ce.get('steered_divergence_median'):.4g} vs bar {ce.get('bar'):.3g} "
           f"(noise {ce.get('noise_ceiling'):.3g})" if ce.get("steered_divergence_median") is not None else "serving-time effect unevaluable")
    text = (f"text probe {tx.get('n_differing')}/{tx.get('n_probes')} differing" if tx.get("n_probes") else "text probe unevaluable")
    dis = "; DISAGREEMENT offline-vs-serving-time (see record)" if rec.get("disagreement") else ""
    return f"{rec.get('verdict')}: hash distinct={rec.get('facts', {}).get('intervention_hash_distinct')}; {dose}; {eff}; {text}{dis}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave-repo", default=None, help="the CODE checkout: its src/ and scripts/ supply the RULE and the proxy client")
    ap.add_argument("--proxy-url", default=None)
    ap.add_argument("--routing-table", default=None)
    ap.add_argument("--cell", default=None, help="the steered cell id (alphasteer)")
    ap.add_argument("--anchor", default="none")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--nominal-dose", type=float, default=None, help="the view's nominal strength, recorded (never a gate)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-logprobs", type=int, default=None, help="default: the driver's MMLU_TOP_LOGPROBS (the server's cap)")
    ap.add_argument("--min-ratio", type=float, default=2.0, help="the instrument's own default")
    ap.add_argument("--abs-floor", type=float, default=1e-3, help="the instrument's own default")
    ap.add_argument("--max-tokens", type=int, default=48, help="the text observation's decode budget (the wave probe's)")
    ap.add_argument("--ratio-against", default=None, help="another rung's record: report this rung's effect size over that rung's")
    ap.add_argument("--dose-artifact", default=None,
                    help="the lane's OFFLINE dose capture ($SHARED/exp/strength_dose/<target>/dose_<target>.json, schema "
                         "strength_dose/1) — D87's fact (B), bound to the served object by sha (D88)")
    ap.add_argument("--staged-id", default=None, help="the staged id of the served object (the artifact's rung key)")
    ap.add_argument("--object-sha256", default=None, help="sha256 of the object.pt this job serves; the artifact rung must match it")
    ap.add_argument("--arm", default="alphasteer", help="the artifact's method block (alphasteer | lora)")
    ap.add_argument("--summary", metavar="RECORD", default=None,
                    help="print ONE human-readable line for an existing record and exit (the wrapper's log line)")
    args = ap.parse_args(argv)
    if args.summary:
        print(summary_line(json.loads(Path(args.summary).read_text())))
        return 0
    missing = [k for k in ("wave_repo", "proxy_url", "routing_table", "cell", "out") if getattr(args, k) is None]
    if missing:
        ap.error("the following arguments are required: " + ", ".join("--" + k.replace("_", "-") for k in missing))

    wave = Path(args.wave_repo).resolve()
    for p in (wave / "src", wave / "scripts", wave / "scripts" / "sdfbase" / "full22"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    out: Dict[str, Any] = {
        "schema": "asscr_low_dose_liveness/1", "rule": "D87 (the coordinator, 2026-09-11): at low dose liveness is NUMERIC; "
        "a text-divergence refusal alone does not stop a cell", "cell": args.cell, "anchor": args.anchor, "tag": args.tag,
        "nominal_dose": args.nominal_dose, "wave_repo": str(wave), "proxy_url": args.proxy_url,
        "routing_table": str(Path(args.routing_table).resolve()), "checks": [], "verdict": "unevaluable",
        "written_at": _now(),
    }

    def record(name: str, status: str, **detail):
        out["checks"].append({"check": name, "status": status, **detail})
        print(f"[low-dose-liveness] {status} {name}: {json.dumps(detail, default=str)[:500]}", flush=True)

    def finish(verdict: str) -> int:
        out["verdict"] = verdict
        out["written_at"] = _now()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"[low-dose-liveness] VERDICT {verdict} -> {args.out}", flush=True)
        return EXIT[verdict]

    try:
        from sisteer.capeval.serving.fidelity import FidelityError, check_logprob_steering_bites
        from sisteer.capeval.serving.proxy import ProxyClientEngine, load_routing_table
        from sisteer.eval.attacks import bon_served as bs
        from sisteer.eval.attacks.served_target import PROBE_PROMPTS, route_for
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_scr_drv", wave / "scripts" / "served_capability_run.py")
        _drv = _ilu.module_from_spec(_spec)
        sys.modules["_scr_drv"] = _drv
        _spec.loader.exec_module(_drv)
    except Exception as e:  # noqa: BLE001 — an unreadable instrument is `unevaluable`, never a pass
        record("instrument", "UNEVALUABLE", error=f"{type(e).__name__}: {e}")
        return finish("unevaluable")

    top_k = int(args.top_logprobs if args.top_logprobs is not None else getattr(_drv, "MMLU_TOP_LOGPROBS", 64))
    out["top_logprobs"] = top_k
    out["thresholds"] = {"min_ratio": args.min_ratio, "abs_floor": args.abs_floor,
                         "rule_source": "sisteer.capeval.serving.fidelity.check_logprob_steering_bites (the instrument's own)"}
    try:
        table = load_routing_table(Path(args.routing_table))
        routes = list(table["routes"])
        by_cell = {r.cell_id: r for r in routes}
        route = route_for(args.cell, routes)
        anchor = route_for(args.anchor, routes)
        twin = by_cell.get(f"{route.cell_id}__lens0") or by_cell.get("none__lens0")
        if twin is None:
            record("routes", "UNEVALUABLE", error=f"steered cell {route.cell_id!r} has no scale-0 twin route "
                                                  "(neither its own __lens0 nor none__lens0) — nothing to measure against")
            return finish("unevaluable")
        client = ProxyClientEngine(args.proxy_url, identity=route.serving_identity)
        client.load(require_virtual_models=[route.virtual_model, anchor.virtual_model, twin.virtual_model])
    except Exception as e:  # noqa: BLE001
        record("routes", "UNEVALUABLE", error=f"{type(e).__name__}: {e}")
        return finish("unevaluable")

    out.update({"virtual_model": route.virtual_model, "anchor_vmodel": anchor.virtual_model, "twin_vmodel": twin.virtual_model,
                "is_lens": bool(bs._is_lens(route)), "serving_identity": route.serving_identity})

    def one_token_logprobs(vmodel: str, prompt: str) -> Dict[int, float]:
        """The first generated token's top-K logprobs, keyed by token id (the driver's reader, its request shape minus the
        teacher-forced seed: a plain greedy one-token request needs no response-prefill declaration on either route kind)."""
        resp = client.chat_completions({"model": vmodel, "messages": [{"role": "user", "content": prompt}],
                                        "temperature": 0.0, "max_tokens": 1, "seed": 0, "logprobs": True,
                                        "top_logprobs": top_k, "return_tokens_as_token_ids": True})
        first = ((resp["choices"][0].get("logprobs") or {}).get("content") or [None])[0]
        if not first:
            raise RuntimeError(f"no logprobs at the first generated position from {vmodel} (the server did not honor "
                               "logprobs/top_logprobs)")
        entries = list(first.get("top_logprobs") or [])
        entries.append({"token": first.get("token"), "logprob": first.get("logprob")})
        by_id: Dict[int, float] = {}
        for e in entries:
            tid = _drv._token_id_of(e.get("token"))
            lp = e.get("logprob")
            if tid is not None and isinstance(lp, (int, float)):
                by_id[int(tid)] = float(lp)
        return by_id

    # --- (A) the intervention identity: distinct from the anchor's, and carried by the steered responses -----------------
    texts: Dict[str, List[str]] = {"steered": [], "twin": [], "anchor": []}
    hashes: Dict[str, set] = {"steered": set(), "anchor": set()}
    try:
        for leg, r in (("steered", route), ("anchor", anchor)):
            resp = client.chat_completions({"model": r.virtual_model, "messages": [{"role": "user", "content": PROBE_PROMPTS[0]}],
                                            "temperature": 0.0, "max_tokens": 8, "seed": 0})
            hashes[leg].add(resp.get("petri_intervention_hash"))
        spec_distinct = route.intervention_hash != anchor.intervention_hash
        served_ok = hashes["steered"] == {route.intervention_hash} and hashes["anchor"] == {anchor.intervention_hash}
        ok_a = bool(spec_distinct and served_ok)
        record("intervention_hash_distinct", "PASS" if ok_a else "FAIL", route_hash=route.intervention_hash,
               anchor_route_hash=anchor.intervention_hash, served_steered=sorted(map(str, hashes["steered"])),
               served_anchor=sorted(map(str, hashes["anchor"])), spec_distinct=spec_distinct, served_matches_spec=served_ok)
    except Exception as e:  # noqa: BLE001
        record("intervention_hash_distinct", "UNEVALUABLE", error=f"{type(e).__name__}: {e}")
        return finish("unevaluable")

    # --- (B) the continuous effect, by the INSTRUMENT'S OWN rule ---------------------------------------------------------
    items: List[Dict[str, Any]] = []
    effect_error: Optional[str] = None
    try:
        for i, prompt in enumerate(PROBE_PROMPTS):
            legs = {"steered": one_token_logprobs(route.virtual_model, prompt),
                    "scale0": one_token_logprobs(twin.virtual_model, prompt),
                    "scale0_repeat": one_token_logprobs(twin.virtual_model, prompt)}
            common = sorted(set(legs["steered"]) & set(legs["scale0"]) & set(legs["scale0_repeat"]))
            if len(common) < 2:
                raise RuntimeError(f"probe {i}: the three legs share only {len(common)} top-{top_k} token(s) — no common "
                                   "support to compare on (raise --top-logprobs and the server's --max-logprobs)")
            items.append({"item_id": f"probe{i:02d}", "letter_ids": common,
                          **{leg: [legs[leg][t] for t in common] for leg in ("steered", "scale0", "scale0_repeat")}})
    except Exception as e:  # noqa: BLE001
        effect_error = f"{type(e).__name__}: {e}"
    diag: Dict[str, Any] = {}
    if effect_error is None:
        try:
            check_logprob_steering_bites({"items": items}, diag, min_ratio=args.min_ratio, abs_floor=args.abs_floor)
        except FidelityError as e:
            effect_error = str(e)
    out["continuous_effect"] = {"diag": diag, "error": effect_error, "n_probes": len(items),
                                "support": "the common top-K token ids of the three legs at the first generated position"}
    if effect_error is not None and not diag:
        record("continuous_effect", "UNEVALUABLE", error=effect_error)
        return finish("unevaluable")
    ok_b = effect_error is None
    record("continuous_effect", "PASS" if ok_b else "FAIL", steered_divergence_median=diag.get("steered_divergence_median"),
           noise_ceiling=diag.get("noise_ceiling"), bar=diag.get("bar"), n_items=diag.get("n_items"), error=effect_error)

    # --- fact (B): the OFFLINE realised dose, bound to the served object by sha --------------------------------------
    rho_emit = bool(list((wave / "src").rglob("*rho_record*")) or list((wave / "src").rglob("*rho_emit*")))
    dose: Dict[str, Any] = {
        "available": False, "source": None, "serving_time_available": False,
        "serving_time_why": ("||edit||/||h|| at serving time needs the lens rho record (SISTEER_LENS_RHO_EMIT / "
                             "sisteer_rho_record), which this code checkout does not carry"
                             + ("" if not rho_emit else " (a rho symbol was found — re-check)")
                             + "; serving the screen on a stack that does would break the identical-instrument rule (D74)"),
        "effect_size_proxy": diag.get("steered_divergence_median"),
        "effect_size_proxy_note": ("a serving-time effect-size PROXY, not a realised dose: a logprob divergence is not linear "
                                   "in the dose the way ||edit|| is"),
    }
    ok_dose: Optional[bool] = None          # None = no artifact to read; True/False = the offline fact's verdict
    if args.dose_artifact:
        dose["artifact"] = str(Path(args.dose_artifact).resolve())
        try:
            art = json.loads(Path(args.dose_artifact).read_text())
            rungs = ((art.get(args.arm) or {}).get("rungs") or {})
            key = args.staged_id or ""
            rung = rungs.get(key)
            if art.get("schema") != "strength_dose/1":
                dose["why"] = f"{args.dose_artifact}: schema {art.get('schema')!r} is not strength_dose/1 — not read"
            elif art.get("target") not in (None, out.get("tag", "").split("_")[0]) and args.tag and not str(args.tag).startswith(str(art.get("target"))):
                dose["why"] = f"{args.dose_artifact}: target {art.get('target')!r} is not this job's — not read"
            elif rung is None:
                dose["why"] = (f"{args.dose_artifact}: no {args.arm} rung {key!r} (rungs: {sorted(rungs)[:8]}) — the offline "
                               "fact is unavailable for this object; the serving-time effect stands in for it")
            elif args.object_sha256 and rung.get("object_sha256") != args.object_sha256:
                dose["why"] = (f"{args.dose_artifact}: rung {key!r} was measured on object "
                               f"{str(rung.get('object_sha256'))[:16]}… but this job serves {args.object_sha256[:16]}… — "
                               "REFUSING to borrow another object's dose. A RE-BAKED ladder is different bytes with different "
                               "layer sets, so it needs its OWN dose capture before its liveness can be established this way "
                               "(D88); until then the serving-time effect is the only fact (B) available here.")
            else:
                med = rung.get("realised_dose_median")
                gate = ((art.get("gate_summary") or {}).get("rungs") or {}).get(f"{args.arm}/{key}")
                nominal_ok = (args.nominal_dose is None or rung.get("nominal_dose") is None
                              or abs(float(rung["nominal_dose"]) - float(args.nominal_dose)) < 1e-12)
                # the SCALING half, recomputed here as an independent cross-check: realised / |nominal| is constant across
                # this arm's rungs (the median is a MAGNITUDE — as_dm0p5 and as_d0p5 read the same — so the sign is the
                # artifact's own signed gate's business, never a signed ratio here)
                per = {}
                for k2, v2 in rungs.items():
                    n2, m2 = v2.get("nominal_dose"), v2.get("realised_dose_median")
                    if isinstance(n2, (int, float)) and isinstance(m2, (int, float)) and n2:
                        per[k2] = m2 / abs(float(n2))
                vals = sorted(per.values())
                ratio_consistent = bool(vals) and (vals[-1] - vals[0]) <= 1e-5 * max(1e-12, vals[-1])
                dose.update({"available": bool(med) and gate is True and nominal_ok, "source": "offline_capture",
                             "realised_dose_median": med, "nominal_dose": rung.get("nominal_dose"),
                             "object_sha256": rung.get("object_sha256"), "staged_id": key,
                             "artifact_ratio_gate": gate, "artifact_gate_all_pass": (art.get("gate_summary") or {}).get("all_pass"),
                             "realised_over_abs_nominal_by_rung": per, "ratio_consistent_rtol_1e-5": ratio_consistent,
                             "effective_layers": (rung.get("effective_layers") or {}).get("effective_layer_count"),
                             "capture": {k3: (art.get("provenance") or {}).get(k3) for k3 in ("code", "gate", "geometry", "h_definition")},
                             "label": ("an OFFLINE measurement (analytic from the object bytes against a captured activation "
                                       "pool under the served rendering) — NOT a serving-time attestation (D88)")})
                if not nominal_ok:
                    dose["why"] = (f"rung {key!r} records nominal {rung.get('nominal_dose')!r} but this job runs "
                                   f"{args.nominal_dose!r} — not read")
                    dose["available"] = False
                elif not med:
                    dose["why"] = f"rung {key!r} realised_dose_median = {med!r}: a ZERO realised dose is not live"
                elif gate is not True:
                    dose["why"] = f"rung {key!r} artifact ratio gate = {gate!r} (not True): the scaling fact does not hold"
                ok_dose = bool(dose["available"]) if (med is not None and gate is not None) else None
                if med == 0 or gate is False:
                    ok_dose = False
        except Exception as e:  # noqa: BLE001 — an unreadable artifact leaves fact (B) to the serving-time effect
            dose["why"] = f"{args.dose_artifact}: unreadable ({type(e).__name__}: {e})"
    else:
        dose["why"] = "no --dose-artifact given; fact (B) falls to the serving-time effect (a labelled proxy)"
    out["realised_dose"] = dose
    record("realised_dose_offline", "PASS" if dose.get("available") else ("FAIL" if ok_dose is False else "UNAVAILABLE"),
           realised_dose_median=dose.get("realised_dose_median"), artifact_ratio_gate=dose.get("artifact_ratio_gate"),
           ratio_consistent=dose.get("ratio_consistent_rtol_1e-5"), source=dose.get("source"), why=dose.get("why"))

    # --- the text probe: an OBSERVATION at low dose, never a gate here ----------------------------------------------
    try:
        n_diff = 0
        rows = []
        for prompt in PROBE_PROMPTS:
            st = client.chat_completions({"model": route.virtual_model, "messages": [{"role": "user", "content": prompt}],
                                          "temperature": 0.0, "max_tokens": args.max_tokens, "seed": 0})["choices"][0]["message"]["content"] or ""
            tw = client.chat_completions({"model": twin.virtual_model, "messages": [{"role": "user", "content": prompt}],
                                          "temperature": 0.0, "max_tokens": args.max_tokens, "seed": 0})["choices"][0]["message"]["content"] or ""
            n_diff += int(st != tw)
            rows.append({"prompt": prompt, "steered": st, "twin": tw, "differs": st != tw})
        out["text_divergence_observation"] = {
            "n_probes": len(PROBE_PROMPTS), "n_differing": n_diff, "rows": rows,
            "meaning": ("the wave instrument's steering_bites probe, RECORDED not gated (D87): at low dose a correct object "
                        "changes no greedy token on a neutral prompt, so 0/5 here is 'live and small' when the numeric "
                        "checks pass — and the dose at which this count leaves 0 is itself worth reporting beside the ladder"),
        }
        record("text_divergence", "OBSERVED", n_differing=n_diff, n_probes=len(PROBE_PROMPTS),
               would_have_refused=(n_diff == 0))
    except Exception as e:  # noqa: BLE001 — the observation is not a gate; its absence is recorded
        out["text_divergence_observation"] = {"error": f"{type(e).__name__}: {e}"}
        record("text_divergence", "UNEVALUABLE", error=f"{type(e).__name__}: {e}")

    # --- the cross-rung scaling half, when another rung's record is given (reported, never a gate) -------------------------
    if args.ratio_against:
        try:
            other = json.loads(Path(args.ratio_against).read_text())
            a = out["continuous_effect"]["diag"].get("steered_divergence_median")
            b = ((other.get("continuous_effect") or {}).get("diag") or {}).get("steered_divergence_median")
            nd_a, nd_b = args.nominal_dose, other.get("nominal_dose")
            out["scaling"] = {"other_record": str(Path(args.ratio_against).resolve()), "other_tag": other.get("tag"),
                              "effect_ratio": (a / b) if (a and b) else None,
                              "nominal_ratio": (nd_a / nd_b) if (nd_a and nd_b) else None,
                              "note": "REPORTED, never a gate: a first-token logprob divergence is not linear in the dose; "
                                      "the exact ||edit|| ratio gate (D51) needs the realised dose, unavailable here"}
            record("scaling", "OBSERVED", **{k: out["scaling"][k] for k in ("effect_ratio", "nominal_ratio")})
        except Exception as e:  # noqa: BLE001
            record("scaling", "UNEVALUABLE", error=f"{type(e).__name__}: {e}")

    # --- the verdict: (A) the intervention distinct, AND fact (B) — the offline realised dose when it binds,
    #     else the serving-time effect as a labelled proxy. A disagreement between the two is flagged, never a stop.
    if ok_dose is None:
        fact_b, fact_b_basis = ok_b, ("the serving-time effect above the engine's own repeat noise (a PROXY: the offline "
                                      "realised dose is unavailable for this object — " + str(dose.get("why")) + ")")
    else:
        fact_b, fact_b_basis = ok_dose, ("the OFFLINE realised dose " + repr(dose.get("realised_dose_median")) +
                                         " with the artifact's signed ratio gate passing, bound to this object by sha (D88) "
                                         "— an offline measurement, NOT a serving-time attestation")
        if ok_dose and not ok_b:
            out["disagreement"] = {
                "what": "the offline realised dose says this object HAS magnitude, but the serving-time effect is not "
                        "distinguishable from the engine's own repeat noise on this standup",
                "verdict_follows": "the offline fact (D87/D88) — the cell is live and the screen proceeds",
                "why_it_matters": "this is the shape a SERVING bug makes (the edit not reaching the served site). If this "
                                  "view's downstream numbers read unsteered, start here.",
                "offline_realised_dose_median": dose.get("realised_dose_median"),
                "serving_time": {k: diag.get(k) for k in ("steered_divergence_median", "noise_ceiling", "bar")},
            }
            record("offline_vs_serving_time", "DISAGREE", **out["disagreement"]["serving_time"])
    out["facts"] = {"intervention_hash_distinct": ok_a, "realised_dose_or_effect": fact_b, "fact_b_basis": fact_b_basis,
                    "serving_time_effect_above_noise": ok_b, "offline_realised_dose_ok": ok_dose}
    if ok_a and fact_b:
        out["basis"] = ("numeric (D87 as amended by D88): the intervention is demonstrably distinct from the anchor's, AND "
                        + fact_b_basis + " — live, whatever the greedy-text probe says")
        return finish("live_numeric")
    failed = [c for c, ok in (("intervention_hash_distinct", ok_a), ("realised_dose_or_effect", fact_b)) if not ok]
    out["basis"] = ("a numeric check FAILED: " + "; ".join(failed) + " (" + fact_b_basis + ")"
                    " — this is genuinely not live, and the screen stops as before (D87)")
    return finish("not_live")


if __name__ == "__main__":
    sys.exit(main())
