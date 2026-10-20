#!/usr/bin/env python
"""LOW-DOSE NUMERIC LIVENESS for the sampled ladder's steered cells — the record writer the launcher runs once per steered
cell after every standup.

Why this exists. The wave instrument proves a served object live with `steering_bites`: greedy TEXT on five neutral probe
prompts against the path-matched scale-0 twin, refusing when NONE of them differs (`fidelity._check_steering_bites`, run
inside every `served_capability_run.py generate` driver and in the standup smoke). At a small enough dose a CORRECT object
changes no greedy token on a neutral prompt — argmax is a step function and a small logit perturbation crosses no boundary —
so that test reports "not live" when steering is live and merely SMALL, and the ladder's stop rule takes the failed positive
control as a STOP. Measured: cluster A 497871 (`f22s-llama8b_it_d0p1-ctl0`) — the derived cell
constv2-mix-full__lora_r4_ce_minus_ctl at nominal 0.1 read 0/5 differing with salt isolation and intervention hashes both
PASS, and the launcher refused the whole ladder on the missing at-cap count.

The rule, as implemented here (the LoRA-side sibling of scripts/alphasteer_f22/screen/asscr_liveness.py, branch
strength-asscreen — same facts, same instrument, no offline dose artifact):
  (A) `intervention_hash_distinct` — the steered route's intervention hash differs from the anchor's AND the served
      responses carry each route's own hash (a silently unchanged intervention has been caught on this project by exactly
      this check);
  (C) `continuous_effect` — a MEASURED non-zero effect at the served site above the engine's own repeat noise, by the
      INSTRUMENT'S OWN rule `fidelity.check_logprob_steering_bites`: three legs (steered / scale-0 twin / the twin AGAIN as
      the noise leg), read as the first generated token's top-K logprobs over the COMMON support of the three legs, on the
      SAME five probe prompts the text check uses (`served_target.PROBE_PROMPTS`); verdict = the steered divergence median
      clears max(abs_floor 1e-3, 2 x the measured repeat-noise ceiling).
  The text probe is RECORDED as an observation (n_probes, n_differing) and never gates anything here.

Fact (B) — the realised dose ||edit|| / ||h|| — is an OFFLINE measurement that the AlphaSteer writer reads from
its lane's dose artifact by object sha. NO such artifact binds a LoRA or a derived (`_minus_ctl`) object, so this writer
takes fact (B) from the serving-time effect (C) and LABELS it a proxy in `realised_dose` (`available: false`, `why`) —
exactly the degradation asscr_liveness.py documents for an unavailable artifact. A later reader must not think the
serving stack attested a realised dose.

Verdicts and exit codes — `live_numeric` (0): (A) and (C) both PASS, the cell is live and small whatever the text probe
says; `not_live` (2): a numeric check FAILED — genuinely not live, the drivers refuse exactly as before; `unevaluable` (3):
the instrument could not be read (no twin route, the proxy down, no common top-K support, ids not honoured) — never
silently a pass, and the launcher leaves the policy env UNSET for that cell.

The record binds to THIS standup: `cell` is the launcher's cell id (the id `served_capability_run.py --cell` and
`smoke_positive_control.py --cell` take) and `routing_table` is the exact path the drivers pass to
low_dose_liveness_policy.consult(); a record from another standup admits nothing there.

Usage (in-job, common_sampled.sh f22s_low_dose_liveness, after `target up` and before the smoke and the drivers):
  python scripts/sdfbase/full22/sampled/lora_liveness.py --proxy-url <.../v1> --routing-table <standup dir>/routing_table.json \
      --cell constv2-mix-full__lora_r4_ce_minus_ctl --tag llama8b_it_d0p1 --view it_d0p1 \
      --out <standup dir>/liveness_low_dose_constv2-mix-full__lora_r4_ce_minus_ctl.json [--top-logprobs 64] [--thinking-on]
  python scripts/sdfbase/full22/sampled/lora_liveness.py --summary <record.json>     # the launcher's one log line
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[4]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22"):   # harness_evidence.py
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "low_dose_liveness/1"
RULE = ("D87 (the coordinator, 2026-09-11): at low dose liveness is NUMERIC — a distinct intervention AND a measured effect "
        "at the served site above the engine's own repeat noise; a text-divergence refusal alone does not stop a cell")
VERDICTS = ("live_numeric", "not_live", "unevaluable")
EXIT = {"live_numeric": 0, "not_live": 2, "unevaluable": 3}
#: the record a relaxed refusal branch consults (low_dose_liveness_policy.POLICY_ENV; the launcher exports it per cell)
POLICY_ENV = "SISTEER_LOW_DOSE_LIVENESS_RECORD"
#: the instrument's own defaults (fidelity.check_logprob_steering_bites)
DEFAULT_MIN_RATIO = 2.0
DEFAULT_ABS_FLOOR = 1e-3
#: vLLM's engine default --max-logprobs, the request cap when neither the flag nor the standup's endpoint.json says otherwise
ENGINE_DEFAULT_MAX_LOGPROBS = 20

#: chat transport: an OpenAI chat payload -> the chat-completion dict (ProxyClientEngine.chat_completions; a stub in tests)
ChatFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class Unevaluable(RuntimeError):
    """The instrument could not be read — recorded as `unevaluable`, never as a pass."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def nominal_dose_of_view(view: Optional[str]) -> Optional[float]:
    """The sampled scope's dose views spell the dose in the view name (make_sampled_configs.DOSE_TOKENS): `d0p1` -> 0.1,
    `it_d1p5` -> 1.5, `dm1` -> -1.0, `d1` -> 1.0. A view without a dose token (`main`, `it`) has none (None) — recorded,
    never guessed."""
    m = re.fullmatch(r"(?:it_)?d(m?)(\d+)(?:p(\d+))?", (view or "").strip())
    if not m:
        return None
    sign, whole, frac = m.groups()
    val = float(f"{whole}.{frac}" if frac else whole)
    return -val if sign else val


def token_id_of(token: Any) -> int:
    """The integer id of a `return_tokens_as_token_ids` logprob entry (`token_id:<int>`). Any other shape means the server
    ignored the flag — the driver's `_token_id_of` refuses the same way, because matching ids by surface string would
    silently misread."""
    if not isinstance(token, str) or not token.startswith("token_id:"):
        raise Unevaluable(f"logprob entry token {token!r} is not in token_id:<int> form — return_tokens_as_token_ids was "
                          "not honored; refusing to match tokens by surface string")
    return int(token[len("token_id:"):])


def first_token_logprobs(chat: ChatFn, vmodel: str, prompt: str, top_k: int) -> Dict[int, float]:
    """The first generated token's top-K logprobs keyed by token id — a plain greedy one-token request (no response
    prefill, so it needs no declaration on either route kind)."""
    resp = chat({"model": vmodel, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 1,
                 "seed": 0, "logprobs": True, "top_logprobs": int(top_k), "return_tokens_as_token_ids": True})
    first = ((resp["choices"][0].get("logprobs") or {}).get("content") or [None])[0]
    if not first:
        raise Unevaluable(f"no logprobs at the first generated position from {vmodel} (the server did not honor "
                          "logprobs/top_logprobs)")
    entries = list(first.get("top_logprobs") or [])
    entries.append({"token": first.get("token"), "logprob": first.get("logprob")})
    by_id: Dict[int, float] = {}
    for e in entries:
        lp = e.get("logprob")
        if isinstance(lp, (int, float)) and not isinstance(lp, bool):
            by_id[token_id_of(e.get("token"))] = float(lp)
    return by_id


def greedy_text(chat: ChatFn, vmodel: str, prompt: str, max_tokens: int):
    """One greedy completion — the request shape of `served_target.probe_fn` (the text check's) — and the served
    intervention hash the proxy stamps on it."""
    resp = chat({"model": vmodel, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                 "max_tokens": int(max_tokens), "seed": 0})
    return (resp["choices"][0]["message"]["content"] or ""), resp.get("petri_intervention_hash")


def measure(chat: ChatFn, *, route, anchor, twin, prompts: Sequence[str], top_k: int, max_tokens: int,
            check_effect: Callable[..., None], fidelity_error: type, min_ratio: float = DEFAULT_MIN_RATIO,
            abs_floor: float = DEFAULT_ABS_FLOOR, chat_text: Optional[ChatFn] = None) -> Dict[str, Any]:
    """The three measurements against a live chat transport: (A) the intervention identity, (C) the continuous effect by
    the instrument's own rule, and the text-divergence OBSERVATION. Returns the record body; raises `Unevaluable` when a
    numeric fact cannot be read (the text observation's own failure is recorded, never raised — it is not a gate)."""
    checks: List[Dict[str, Any]] = []

    def record(name: str, status: str, **detail):
        checks.append({"check": name, "status": status, **detail})
        print(f"[low-dose-liveness] {status} {name}: {json.dumps(detail, default=str)[:500]}", flush=True)

    # --- the text probe (the wave's steering_bites shape: steered vs the scale-0 twin, greedy, PROBE_MAX_TOKENS) — an
    #     OBSERVATION; the anchor leg rides along so the served hashes of fact (A) come from the same requests
    rows: List[Dict[str, Any]] = []
    hashes: Dict[str, set] = {"steered": set(), "anchor": set()}
    text_error: Optional[str] = None
    try:
        for prompt in prompts:
            st, st_h = greedy_text(chat_text or chat, route.virtual_model, prompt, max_tokens)
            tw, _ = greedy_text(chat_text or chat, twin.virtual_model, prompt, max_tokens)
            an, an_h = greedy_text(chat_text or chat, anchor.virtual_model, prompt, max_tokens)
            hashes["steered"].add(st_h)
            hashes["anchor"].add(an_h)
            rows.append({"prompt": prompt, "steered": st, "twin": tw, "anchor": an, "differs": st != tw,
                         "steered_ne_anchor": st != an})
    except Exception as e:  # noqa: BLE001 — recorded; the numeric facts below decide
        text_error = f"{type(e).__name__}: {e}"
    n_diff = sum(int(r["differs"]) for r in rows)
    text_obs: Dict[str, Any] = {
        "n_probes": len(prompts), "n_completed": len(rows), "n_differing": n_diff,
        "n_steered_ne_anchor": sum(int(r["steered_ne_anchor"]) for r in rows), "rows": rows, "error": text_error,
        "would_have_refused": (n_diff == 0) if text_error is None else None,
        "meaning": ("the wave instrument's steering_bites probe (fidelity._check_steering_bites), RECORDED not gated (D87): at "
                    "low dose a correct object changes no greedy token on a neutral prompt, so 0/5 here is 'live and small' "
                    "when the numeric checks pass — and the dose at which this count leaves 0 is itself worth reporting"),
    }
    record("text_divergence", "OBSERVED" if text_error is None else "UNEVALUABLE", n_differing=n_diff,
           n_probes=len(prompts), would_have_refused=text_obs["would_have_refused"], error=text_error)

    # --- (A) the intervention identity: distinct from the anchor's by spec, and carried by the served responses
    if text_error is not None:
        raise Unevaluable(f"intervention_hash_distinct needs the served hashes, and the text legs failed: {text_error}")
    spec_distinct = route.intervention_hash != anchor.intervention_hash
    served_ok = hashes["steered"] == {route.intervention_hash} and hashes["anchor"] == {anchor.intervention_hash}
    ok_a = bool(spec_distinct and served_ok)
    hash_detail = {"route_hash": route.intervention_hash, "anchor_route_hash": anchor.intervention_hash,
                   "served_steered": sorted(map(str, hashes["steered"])), "served_anchor": sorted(map(str, hashes["anchor"])),
                   "spec_distinct": spec_distinct, "served_matches_spec": served_ok}
    record("intervention_hash_distinct", "PASS" if ok_a else "FAIL", **hash_detail)

    # --- (C) the continuous effect, by the INSTRUMENT'S OWN rule, on the SAME prompts
    items: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        legs = {"steered": first_token_logprobs(chat, route.virtual_model, prompt, top_k),
                "scale0": first_token_logprobs(chat, twin.virtual_model, prompt, top_k),
                "scale0_repeat": first_token_logprobs(chat, twin.virtual_model, prompt, top_k)}
        common = sorted(set(legs["steered"]) & set(legs["scale0"]) & set(legs["scale0_repeat"]))
        if len(common) < 2:
            raise Unevaluable(f"probe {i}: the three legs share only {len(common)} top-{top_k} token id(s) — no common "
                              "support to compare on (raise --top-logprobs and the standup's --max-logprobs)")
        items.append({"item_id": f"probe{i:02d}", "letter_ids": common,
                      **{leg: [legs[leg][t] for t in common] for leg in ("steered", "scale0", "scale0_repeat")}})
    diag: Dict[str, Any] = {}
    effect_error: Optional[str] = None
    try:
        check_effect({"items": items}, diag, min_ratio=min_ratio, abs_floor=abs_floor)
    except fidelity_error as e:
        effect_error = str(e)
    if effect_error is not None and not diag:
        # the rule refused the SHAPE of the evidence (a malformed item), not the effect — nothing was measured
        raise Unevaluable(f"continuous_effect: {effect_error}")
    ok_c = effect_error is None
    record("continuous_effect", "PASS" if ok_c else "FAIL", steered_divergence_median=diag.get("steered_divergence_median"),
           noise_ceiling=diag.get("noise_ceiling"), bar=diag.get("bar"), n_items=diag.get("n_items"), error=effect_error)

    return {
        "intervention_hash_distinct": ok_a, "intervention_hash_detail": hash_detail,
        "continuous_effect": {"diag": diag, "error": effect_error, "n_probes": len(items),
                              "support": "the common top-K token ids of the three legs at the first generated position"},
        "text_divergence_observation": text_obs, "checks": checks,
        "facts": {"intervention_hash_distinct": ok_a, "serving_time_effect_above_noise": ok_c},
    }


def decide(rec: Dict[str, Any]) -> str:
    """The verdict from the two numeric facts (pure; the record's `facts` block is the input). `unevaluable` is decided by
    the caller when a fact could not be read at all."""
    facts = rec.get("facts") or {}
    ok_a, ok_c = facts.get("intervention_hash_distinct"), facts.get("serving_time_effect_above_noise")
    if ok_a is None or ok_c is None:
        return "unevaluable"
    return "live_numeric" if (ok_a and ok_c) else "not_live"


def basis_of(rec: Dict[str, Any], verdict: str) -> str:
    facts = rec.get("facts") or {}
    fact_b = ("the serving-time first-token logprob effect above the engine's own repeat noise (a PROXY for D87's realised "
              "dose: no offline dose artifact binds a LoRA / derived object — " + str((rec.get("realised_dose") or {}).get("why")) + ")")
    if verdict == "live_numeric":
        return ("numeric (D87): the intervention is demonstrably distinct from the anchor's, AND " + fact_b +
                " — live, whatever the greedy-text probe says")
    if verdict == "not_live":
        failed = [c for c, ok in (("intervention_hash_distinct", facts.get("intervention_hash_distinct")),
                                  ("serving_time_effect_above_noise", facts.get("serving_time_effect_above_noise"))) if not ok]
        return ("a numeric check FAILED: " + "; ".join(failed) + " (" + fact_b + ") — this is genuinely not live, and the "
                "cell refuses exactly as before (D87)")
    return "the instrument could not be read — no numeric fact established; the text-divergence gate stands unchanged"


def summary_line(rec: Mapping[str, Any]) -> str:
    """The one line the launcher logs per cell: `<verdict> median=<steered divergence> bar=<bar> n_differing=<k>/<n> ...`."""
    ce = (rec.get("continuous_effect") or {}).get("diag") or {}
    tx = rec.get("text_divergence_observation") or {}

    def g(v, fmt):
        return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) else "None"

    n_diff = tx.get("n_differing") if tx.get("error") is None and tx.get("n_probes") else "?"
    return (f"{rec.get('verdict')} median={g(ce.get('steered_divergence_median'), '.4g')} bar={g(ce.get('bar'), '.3g')} "
            f"n_differing={n_diff}/{tx.get('n_probes', '?')} hash_distinct={rec.get('intervention_hash_distinct')} "
            f"noise={g(ce.get('noise_ceiling'), '.3g')} nominal_dose={rec.get('nominal_dose')}")


def _top_k_default(routing_table: Path) -> tuple:
    """The request's top-K when --top-logprobs is not given: the standup's own cap from endpoint.json beside the routing
    table (the authoritative stamp), else vLLM's engine default. Returns (k, source)."""
    ep = Path(routing_table).resolve().parent / "endpoint.json"
    try:
        v = json.loads(ep.read_text()).get("max_logprobs")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v, f"endpoint.json max_logprobs ({ep})"
    except Exception:  # noqa: BLE001 — an absent or unreadable stamp falls to the engine default, recorded as such
        pass
    return ENGINE_DEFAULT_MAX_LOGPROBS, "vLLM engine default (--max-logprobs unset on the standup; no endpoint.json read)"


def main(argv=None, *, client_factory: Optional[Callable[..., Any]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-url", default=None)
    ap.add_argument("--routing-table", default=None, help="the standup's routing_table.json — the exact path the drivers pass")
    ap.add_argument("--cell", default=None, help="the launcher's steered cell id (e.g. constv2-mix-full__lora_r4_ce_minus_ctl)")
    ap.add_argument("--anchor", default="none")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--view", default=None, help="the bank view (it_d0p1 …): the nominal dose is read off it, recorded, never a gate")
    ap.add_argument("--nominal-dose", type=float, default=None, help="overrides the dose read off --view")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-logprobs", type=int, default=None,
                    help="default: the standup's endpoint.json max_logprobs beside the routing table, else the engine default 20")
    ap.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO, help="the instrument's own default")
    ap.add_argument("--abs-floor", type=float, default=DEFAULT_ABS_FLOOR, help="the instrument's own default")
    ap.add_argument("--max-tokens", type=int, default=None, help="the text observation's decode budget (default: the wave probe's)")
    ap.add_argument("--thinking-on", action="store_true",
                    help="the standup was `mt_serving_up.py --thinking-on`: every LENS-route request carries the proxy's "
                         "declaration key (the bare anchor must not) — the smoke's rule")
    ap.add_argument("--summary", metavar="RECORD", default=None, help="print ONE line for an existing record and exit")
    args = ap.parse_args(argv)
    if args.summary:
        print(summary_line(json.loads(Path(args.summary).read_text())))
        return 0
    missing = [k for k in ("proxy_url", "routing_table", "cell", "out") if getattr(args, k) is None]
    if missing:
        ap.error("the following arguments are required: " + ", ".join("--" + k.replace("_", "-") for k in missing))

    nominal = args.nominal_dose if args.nominal_dose is not None else nominal_dose_of_view(args.view)
    out: Dict[str, Any] = {
        "schema": SCHEMA, "rule": RULE, "cell": args.cell, "anchor": args.anchor, "tag": args.tag, "view": args.view,
        "nominal_dose": nominal, "proxy_url": args.proxy_url, "routing_table": str(Path(args.routing_table).resolve()),
        "thinking_on": bool(args.thinking_on), "checks": [], "verdict": "unevaluable", "written_at": _now(),
        "realised_dose": {
            "available": False, "source": None,
            "why": ("D87's fact (B), the realised dose ||edit||/||h||, is an OFFLINE measurement (D88) read from a dose artifact "
                    "bound by object sha; no such artifact binds a LoRA / derived object, and it is not measurable at serving "
                    "time on this code sha — fact (B) falls to the serving-time effect, a labelled proxy"),
            "effect_size_proxy": None,
            "effect_size_proxy_note": ("a serving-time effect-size PROXY, not a realised dose: a logprob divergence is not "
                                       "linear in the dose the way ||edit|| is")},
    }

    def finish(verdict: str) -> int:
        out["verdict"] = verdict
        out["basis"] = basis_of(out, verdict)
        out["written_at"] = _now()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"[low-dose-liveness] VERDICT {verdict} -> {args.out}", flush=True)
        return EXIT[verdict]

    def unevaluable(where: str, err: str) -> int:
        out["checks"].append({"check": where, "status": "UNEVALUABLE", "error": err})
        print(f"[low-dose-liveness] UNEVALUABLE {where}: {err[:600]}", flush=True)
        return finish("unevaluable")

    try:
        from sisteer.capeval.serving.fidelity import FidelityError, check_logprob_steering_bites
        from sisteer.capeval.serving.proxy import THINKING_ON_KEY, ProxyClientEngine, load_routing_table
        from sisteer.eval.attacks import bon_served as bs
        from sisteer.eval.attacks.served_target import PROBE_MAX_TOKENS, PROBE_PROMPTS, route_for
    except Exception as e:  # noqa: BLE001 — an unreadable instrument is `unevaluable`, never a pass
        return unevaluable("instrument", f"{type(e).__name__}: {e}")

    if args.top_logprobs is not None:
        top_k, top_src = int(args.top_logprobs), "--top-logprobs"
    else:
        top_k, top_src = _top_k_default(Path(args.routing_table))
    out.update({"top_logprobs": top_k, "top_logprobs_source": top_src, "probe_prompts": list(PROBE_PROMPTS),
                "thresholds": {"min_ratio": args.min_ratio, "abs_floor": args.abs_floor,
                               "rule_source": "sisteer.capeval.serving.fidelity.check_logprob_steering_bites (the instrument's own)"}})
    max_tokens = args.max_tokens if args.max_tokens is not None else PROBE_MAX_TOKENS

    try:
        table = load_routing_table(Path(args.routing_table))
        routes = list(table["routes"])
        by_cell = {r.cell_id: r for r in routes}
        route = route_for(args.cell, routes)
        anchor = route_for(args.anchor, routes)
        if not bs._is_lens(route):
            return unevaluable("routes", f"cell {args.cell!r} (route {route.cell_id!r}) is not lens-served "
                                         f"({route.serving_identity}) — there is no scale-0 twin to measure against, and the "
                                         "text-divergence check does not apply to it either")
        twin = by_cell.get(f"{route.cell_id}__lens0") or by_cell.get("none__lens0")
        if twin is None:
            return unevaluable("routes", f"steered cell {route.cell_id!r} has no scale-0 twin route (neither its own __lens0 "
                                         "nor none__lens0) — nothing to measure against")
        # the greedy text probes are the bites shape and are labelled at the source (X-Tinycap-Task: probe/bites), every request's
        # wire bytes recorded to <out stem>.request_evidence.jsonl beside this record — the sidecar the sampled resolved-sampler read
        # hands to the reader (common_sampled.sh f22s_record_resolved_sampler), so these probes are carved by their tag, never read as
        # unclassified greedy traffic. The 1-token numeric probes carry no kind label.
        from harness_evidence import HarnessEvidence, evidence_path_for
        evidence = HarnessEvidence(evidence_path_for(Path(args.out)))
        out["request_evidence"] = str(evidence.path); out["kind_tag"] = "probe/bites"
        factory = client_factory or (lambda url, identity: ProxyClientEngine(url, identity=identity, on_request=evidence.on_request))
        client = factory(args.proxy_url, route.serving_identity)
        client.load(require_virtual_models=[route.virtual_model, anchor.virtual_model, twin.virtual_model])
    except Exception as e:  # noqa: BLE001
        return unevaluable("routes", f"{type(e).__name__}: {e}")
    out.update({"harness_cell_id": route.cell_id, "virtual_model": route.virtual_model, "anchor_vmodel": anchor.virtual_model,
                "twin": twin.cell_id, "twin_vmodel": twin.virtual_model, "serving_identity": route.serving_identity})

    # the thinking regime (the smoke's rule): under a thinking-ON standup every LENS-route request carries the proxy's
    # declaration key and the bare anchor must not; OFF: the bodies are byte-identical to the wave probe's
    lens_vmodels = {r.virtual_model for r in routes if bs._is_lens(r)}

    def chat_h(body: Mapping[str, Any], headers: Mapping[str, Any]) -> Mapping[str, Any]:
        if args.thinking_on and body.get("model") in lens_vmodels:
            body = {**body, THINKING_ON_KEY: True}
        return client.chat_completions(body, headers=dict(headers))

    def chat(body: Mapping[str, Any]) -> Mapping[str, Any]:      # the 1-token numeric probes: no kind label
        return chat_h(body, {})

    chat_text = evidence.labelled(chat_h, "bites")                 # the greedy text probes: probe/bites, sample numbered per route

    try:
        out.update(measure(chat, chat_text=chat_text, route=route, anchor=anchor, twin=twin, prompts=PROBE_PROMPTS, top_k=top_k,
                           max_tokens=max_tokens, check_effect=check_logprob_steering_bites, fidelity_error=FidelityError,
                           min_ratio=args.min_ratio, abs_floor=args.abs_floor))
    except Unevaluable as e:
        return unevaluable("measure", str(e))
    except Exception as e:  # noqa: BLE001 — a transport or server failure is not a verdict
        return unevaluable("measure", f"{type(e).__name__}: {e}")
    out["realised_dose"]["effect_size_proxy"] = (out["continuous_effect"].get("diag") or {}).get("steered_divergence_median")
    out["facts"]["realised_dose_or_effect"] = out["facts"]["serving_time_effect_above_noise"]
    out["facts"]["fact_b_basis"] = "the serving-time effect above the engine's own repeat noise (a labelled PROXY; see realised_dose.why)"
    return finish(decide(out))


if __name__ == "__main__":
    sys.exit(main())
