#!/usr/bin/env python
"""full22: the REPRESENTATIVE LOAD CHECK of a standup — what makes a provisional serving budget
usable without a fresh maximum-budget probe (the sdfbase-llama8b lane).

After a standup passed readiness (kv_coverage_problem at the identity geometry) and attestation
(the workers' (S, L) == the resolved one), this fires S CONCURRENT requests at the server, each with
a prompt whose RENDERED length is the leg's p99 prompt (or less, when the window leaves less room) and
max_tokens = the RUN's cap, and REQUIRES every response to finish with finish_reason in {stop, length},
no 400 and no 5xx, and the server alive afterwards. It writes load_check.json beside endpoint.json
({S, L, cap, probe, n_ok, finish_reasons, wall_s, ok, verdict}). A failure here is a MEASURED limit at
that window (risk 12), reported as such — never a silent shrink of S.

Review round 1 (2026-09-10) fixed two defects of the first cut:
  * P1-1 the fullconst launchers send the full SI as the system message, and the production proxy
    REFUSES a registered prefix on a non-prompted route (si_prefix_unexpected). With --system-si the
    probe goes to the PROMPTED route whose registered prefix hash matches (from the standup's routing
    table); `none` is used for prefix-free probes only.
  * P1-2 the probe no longer combines the table's pooled p99 prompt with the table's cap regardless of
    the window (multi-turn 7,496 + 2,048 at 8,704 and Petri 17,714 + 1,024 at 16,384 were guaranteed
    400s read as "measured limits"): the cap is the RUN's (--cap, or the config's per eval, else the
    table's), the prompt is a RENDERED count (system prefix + chat scaffold + user filler) sized to
    min(p99_prompt, L - cap), and a probe the window cannot carry at all is INCOMPATIBLE (exit 4, before
    any request), naming the geometry the shared budget resolves for the intended probe — never a
    predictable 400 diagnosed as capacity. --preflight runs exactly that validation with no server.

  python scripts/sdfbase/full22/load_check.py --endpoint <out>/endpoint.json --eval families_served:short \\
      --config <wave config> --model-key llama8b [--cap 4096] [--prompt-tokens 762] [--system-si si_constitution_full] \\
      [--concurrency S] [--out load_check.json]
  python scripts/sdfbase/full22/load_check.py --preflight --config <cfg> --eval multiturn_attacks --model-key llama8b \\
      [--max-model-len L] [--cap N] [--system-si ID] --out preflight.json

Exit 0 = passed; 3 = the server refused / failed at this width (the measured limit); 4 = the probe is
INCOMPATIBLE with the window (refused before any request; the record names the geometry that fits);
2 = a usage / record problem.

TAGGED AT THE SOURCE (recorder-load-check-kinds, 2026-09-13): every request carries the tinycap driver's
kind label — ``X-Tinycap-Task: probe/load_check`` (+ sample + a fresh request id) — and is recorded from its
exact wire bytes to ``<out stem>.request_evidence.jsonl`` beside --out (``harness_evidence.py``; --request-evidence
overrides), so ``record_resolved_sampler.py --probe-expect kinds --probe-tags <that file>`` classifies the
load check by a DECLARED identity: it fires at the RUN's cap, the leg's own client cap, and so shares the client
class key by construction (job 97376: 46 greedy requests inside a 632-request sampled client class read as a
false DIVERGED). The request BODY is byte-identical to before; only headers and the sidecar were added.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:      # harness_evidence.py, a sibling
    sys.path.insert(0, str(Path(__file__).resolve().parent))

#: the kind label the load check's requests carry (contracts.PROBE_KIND_LOAD_CHECK; pinned by test)
LOAD_CHECK_KIND = "load_check"

OK_FINISH = ("stop", "length")
EXIT_OK, EXIT_USAGE, EXIT_MEASURED_LIMIT, EXIT_INCOMPATIBLE = 0, 2, 3, 4
#: eval -> the config key that holds THAT leg's cap (the run's, never the table's)
CAP_KEY_OF_EVAL = {
    # the multi-turn leg's cap is its OWN key when the config names one (the sdfbase-llama8b lane's caps:
    # attacks.multiturn_target_max_new_tokens, read by jailbreak_run._multiturn_target_max_new_tokens); a tuple of
    # candidates is tried in order, so a config without the key keeps the shared cap
    "multiturn_attacks": ("attacks", ("multiturn_target_max_new_tokens", "target_max_new_tokens")),
    "families_served": ("attacks", "static_max_new_tokens"),
    "families_prompted": ("bon", "max_new_tokens"),
    "bon": ("bon", "max_new_tokens"),
    # the judged controls' cap rides its OWN key since 2026-09-11 (attacks.controls_max_new_tokens; cell_controls.controls_cap)
    # beside the multi-turn cap attacks.target_max_new_tokens (4,096 on every wave config): the controls standup's probe is
    # sized from the controls' cap, else the shared cap exactly as before for a config without the key
    "capability": ("attacks", ("controls_max_new_tokens", "target_max_new_tokens")),
}


def read_endpoint(path: Path) -> dict:
    ep = json.loads(Path(path).read_text())
    for k in ("proxy_url", "max_model_len", "max_num_seqs", "model", "model_revision"):
        if k not in ep:
            raise SystemExit(f"[load-check] {path}: endpoint.json lacks {k!r} — the standup's record is the "
                             "only source of the served geometry; refusing to guess")
    return ep


def _routes(ep: dict) -> list:
    table = ep.get("routing_table")
    if not table or not Path(table).is_file():
        return []
    from sisteer.capeval.serving.proxy import load_routing_table
    return list(load_routing_table(Path(table))["routes"])


def probe_vmodel(ep: dict, system_si: Optional[str], system_text: Optional[str]) -> str:
    """The vmodel the probe hits: with a system prefix, the PROMPTED route whose registered prefix hash is
    the prefix's (the proxy refuses the prefix on any other route); without one, `none`."""
    routes = _routes(ep)
    if system_si:
        from sisteer.capeval.manifest import sha256_hex16
        h = sha256_hex16(system_text.encode("utf-8"))
        hits = [r for r in routes if r.cell_id.startswith("prompted__") and r.si_prefix_hash == h]
        if not hits:
            raise SystemExit(f"[load-check] --system-si {system_si}: no prompted route in {ep.get('routing_table')} "
                             f"registers a prefix with hash {h} (routes: {[r.cell_id for r in routes]}) — the "
                             "proxy refuses a registered prefix on any other route; the view does not serve this SI")
        return hits[0].virtual_model
    for r in routes:
        if r.cell_id == "none":
            return r.virtual_model
    for vm in ep.get("served_models") or []:
        if vm.endswith("__base"):
            return vm
    raise SystemExit("[load-check] cannot find the `none` route in the routing table / served_models")


def table_entry(eval_name: str):
    """The shared budget table's entry for ``eval`` or ``eval:task`` (geometry_for parses both forms)."""
    from sisteer.capeval.serving.geometry import geometry_for
    return geometry_for(eval_name)


def cap_for(eval_name: str, cfg: Optional[dict], explicit: Optional[int], *, attack: Optional[str] = None) -> tuple:
    """(cap, source): --cap, else the RUN's config value for the leg, else the table's requested cap. A config value that
    is a PER-FAMILY map (attacks.multiturn_target_max_new_tokens on the sdfbase-llama8b lane: {crescendo, pair,
    actorattack} -> cap) needs ``attack`` — the family, or the comma list of families, the standup serves; the probe
    takes the MAX over them (the most demanding probe the window must carry) and REFUSES a family the map does not
    name, or no family at all — never a default."""
    if explicit is not None:
        return int(explicit), "flag:--cap"
    base = eval_name.split(":", 1)[0]
    if cfg is not None and base in CAP_KEY_OF_EVAL:
        section, key = CAP_KEY_OF_EVAL[base]
        blk = cfg.get(section) or {}
        for k in (key if isinstance(key, tuple) else (key,)):
            if k in blk:
                v = blk[k]
                if isinstance(v, dict):
                    fams = [a for a in (attack or "").split(",") if a]
                    if not fams:
                        raise SystemExit(f"[load-check] {section}.{k} is a per-family map {sorted(v)}: pass --attack "
                                         "<family[,family...]> (the family(ies) this standup serves)")
                    missing = [a for a in fams if a not in v]
                    if missing:
                        raise SystemExit(f"[load-check] {section}.{k} names no cap for {missing} (families: {sorted(v)}) — "
                                         "a family the config does not name cannot run")
                    caps = {a: int(v[a]) for a in fams}
                    top = max(caps, key=lambda a: caps[a])
                    return caps[top], f"config:{section}.{k}[{top}]" + (f" (max over {','.join(fams)})" if len(fams) > 1 else "")
                return int(v), f"config:{section}.{k}"
    b = table_entry(eval_name).basis
    cap = getattr(b, "requested_max_tokens", None)
    if cap is None:
        raise SystemExit(f"[load-check] no cap for {eval_name!r}: pass --cap or --config")
    return int(cap), "table:basis.requested_max_tokens"


def _messages(system: Optional[str], user: str) -> list:
    """The chat the probe renders and sends: [system (when given), user]."""
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]


def rendered_len(tok, system: Optional[str], user: str) -> int:
    """The SERVED render's token count: the chat template over the messages + the generation header
    (transformers returns a list or a BatchEncoding depending on the version — both read)."""
    ids = tok.apply_chat_template(_messages(system, user), add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys") or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def filler_for(tok, system: Optional[str], target_rendered: int) -> tuple:
    """A user text whose RENDERED prompt (system + scaffold + user) is <= target_rendered and as close
    as the tokenizer allows; returns (text, rendered_tokens)."""
    word = "lorem "
    base = rendered_len(tok, system, "")
    room = max(0, target_rendered - base)
    if room == 0:
        return "", base
    per = max(1, len(tok(word * 64, add_special_tokens=False)["input_ids"]) / 64)
    n_words = max(1, int(room / per))
    text = word * n_words
    n = rendered_len(tok, system, text)
    while n > target_rendered and n_words > 1:
        n_words = max(1, n_words - max(1, int((n - target_rendered) / per) + 1))
        text = word * n_words
        n = rendered_len(tok, system, text)
    return text, n


def plan_probe(*, L: int, eval_name: str, cap: int, cap_source: str, tok, system: Optional[str],
               system_si: Optional[str], prompt_tokens: Optional[int], model_key: Optional[str]) -> dict:
    """The probe, SIZED TO THE WINDOW: rendered prompt = min(p99_prompt-or-flag, L - cap), or an
    INCOMPATIBLE verdict naming the geometry the shared budget resolves for the intended probe."""
    b = table_entry(eval_name).basis
    p99 = int(prompt_tokens) if prompt_tokens is not None else int(getattr(b, "p99_prompt", 0) or 0)
    scaffold = rendered_len(tok, system, "")            # system prefix (if any) + the chat scaffold
    intended = max(p99, scaffold)                       # the p99 prompt is a rendered figure; never below the scaffold
    room = L - cap
    out = {"eval": eval_name, "L": L, "cap": cap, "cap_source": cap_source, "p99_prompt": p99,
           "system_si": system_si, "scaffold_tokens": scaffold, "intended_prompt_tokens": intended,
           "room_for_prompt": room}
    if room < scaffold + 1:
        needed = scaffold + max(1, min(p99, scaffold + 8)) + cap
        out.update(compatible=False, target_prompt_tokens=None,
                   why=(f"cap {cap} ({cap_source}) leaves {room} tokens of window {L} for a prompt whose scaffold "
                        f"alone is {scaffold} — no probe can fit; not a capacity fact"),
                   needed_geometry=_needed_geometry(eval_name, needed, model_key))
        return out
    target = min(intended, room)
    text, n = filler_for(tok, system, target)
    out.update(compatible=True, target_prompt_tokens=target, rendered_prompt_tokens=n, user_text=text,
               shrunk_from_p99=(intended > room),
               needed_geometry=(_needed_geometry(eval_name, intended + cap, model_key) if intended > room else None))
    return out


def _needed_geometry(eval_name: str, needed_tokens: int, model_key: Optional[str]) -> dict:
    """The geometry the SHARED budget resolves for a window that carries `needed_tokens` — S follows from
    budget_for(model_key), never hand-picked; recorded so a predictable 400 is never read as capacity."""
    from sisteer.capeval.serving.geometry import next_block, resolve_geometry
    L = next_block(int(needed_tokens))
    rec = {"needed_tokens": int(needed_tokens), "max_model_len": L}
    if model_key:
        try:
            g = resolve_geometry(eval_name, max_model_len=L, model_key=model_key)
            rec.update(max_num_seqs=g.max_num_seqs, budget=g.budget_tokens, budget_source=g.budget_source,
                       identity=g.is_identity)
        except Exception as e:  # noqa: BLE001 — recorded, never swallowed
            rec["resolve_error"] = f"{type(e).__name__}: {e}"
    return rec


async def _fire(post, vmodel: str, *, system, user: str, cap: int, S: int, evidence=None) -> list:
    """S concurrent greedy requests through ``post(body, headers)``; with ``evidence`` (a
    ``harness_evidence.HarnessEvidence``) each carries the ``probe/load_check`` labels and is recorded
    from the exact bytes the poster serialises (``json.dumps(body)``) before it is sent."""
    async def one(i):
        body = {"model": vmodel, "messages": _messages(system, user), "max_tokens": int(cap), "temperature": 0.0,
                "seed": i}
        headers = evidence.headers(LOAD_CHECK_KIND, vmodel) if evidence is not None else {}
        if evidence is not None:
            evidence.record(json.dumps(body).encode("utf-8"), headers)
        t0 = time.time()
        try:
            status, text = await post(body, headers)
        except Exception as e:  # noqa: BLE001 — recorded as a failed request
            return {"i": i, "status": None, "error": f"{type(e).__name__}: {e}", "latency_s": time.time() - t0}
        rec = {"i": i, "status": status, "latency_s": round(time.time() - t0, 3)}
        if status == 200:
            try:
                d = json.loads(text)
                rec["finish_reason"] = d["choices"][0].get("finish_reason")
                rec["prompt_tokens"] = (d.get("usage") or {}).get("prompt_tokens")
                rec["completion_tokens"] = (d.get("usage") or {}).get("completion_tokens")
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"unparseable 200: {type(e).__name__}: {e}"
        else:
            rec["error"] = str(text)[:400]
        return rec
    return list(await asyncio.gather(*(one(i) for i in range(S))))


def aiohttp_poster(proxy_url: str, timeout_s: float):
    import aiohttp

    async def post(body, headers=None):
        # the body goes on the wire as json.dumps(body) — the same bytes _fire digested into the evidence record
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{proxy_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
                                    headers={**dict(headers or {}), "Content-Type": "application/json"}) as resp:
                return resp.status, await resp.text()
    return post


def server_alive(ep: dict):
    pid = ep.get("pid")
    if not pid:
        return None
    try:
        os.kill(int(pid), 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run(args, *, post=None) -> int:
    from sisteer.models import load_tokenizer

    cfg = None
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.preflight:
        if args.max_model_len is None:
            if not args.model_key:
                raise SystemExit("[load-check] --preflight needs --max-model-len or --model-key (to resolve the table window)")
            from sisteer.capeval.serving.geometry import resolve_geometry
            g = resolve_geometry(args.eval, model_key=args.model_key)
            L, S = g.max_model_len, g.max_num_seqs
        else:
            L = int(args.max_model_len); S = None
        model = cfg["model"] if cfg else None
        rev = cfg.get("model_revision") if cfg else None
        if model is None:
            raise SystemExit("[load-check] --preflight needs --config (the model pins for the tokenizer)")
        ep = None
    else:
        ep = read_endpoint(Path(args.endpoint))
        L, S = int(ep["max_model_len"]), int(ep["max_num_seqs"])
        model, rev = ep["model"], ep["model_revision"]
    if args.concurrency:
        S = int(args.concurrency)
    if not args.eval:
        raise SystemExit("[load-check] --eval names the leg (its cap key and its p99 prompt)")
    cap, cap_source = cap_for(args.eval, cfg, args.cap, attack=args.attack)
    tok = load_tokenizer(model, revision=rev)
    system = None
    if args.system_si:
        from sisteer.data.instructions import BY_ID
        system = BY_ID[args.system_si].text
    probe = plan_probe(L=L, eval_name=args.eval, cap=cap, cap_source=cap_source, tok=tok, system=system,
                       system_si=args.system_si, prompt_tokens=args.prompt_tokens, model_key=args.model_key)
    out_path = Path(args.out) if args.out else (Path(args.endpoint).parent / "load_check.json" if args.endpoint
                                                else Path("load_check.json"))
    base = {"schema": "load_check/2", "S": S, "L": L, "cap": cap, "eval": args.eval, "system_si": args.system_si,
            "probe": {k: v for k, v in probe.items() if k != "user_text"}, "preflight": bool(args.preflight),
            "endpoint": args.endpoint, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not probe["compatible"]:
        rec = {**base, "ok": False, "n_requests": 0, "verdict": ("INCOMPATIBLE: " + probe["why"] +
               f"; the intended probe needs {probe['needed_geometry']}")}
        out_path.write_text(json.dumps(rec, indent=2, default=str))
        print(f"[load-check] INCOMPATIBLE (no request sent): {probe['why']} -> {out_path}", flush=True)
        return EXIT_INCOMPATIBLE
    if args.preflight:
        rec = {**base, "ok": True, "n_requests": 0,
               "verdict": (f"PREFLIGHT OK: a {probe['rendered_prompt_tokens']}-token rendered prompt + cap {cap} fits "
                           f"window {L}" + (" (the p99 prompt was shrunk to the room the window leaves; the intended "
                                            f"probe needs {probe['needed_geometry']})" if probe["shrunk_from_p99"] else ""))}
        out_path.write_text(json.dumps(rec, indent=2, default=str))
        print(f"[load-check] {rec['verdict']} -> {out_path}", flush=True)
        return EXIT_OK
    vmodel = args.vmodel or probe_vmodel(ep, args.system_si, system)
    post = post or aiohttp_poster(ep["proxy_url"], float(args.timeout_s))
    from harness_evidence import HarnessEvidence, evidence_path_for
    evidence = HarnessEvidence(Path(args.request_evidence) if args.request_evidence else evidence_path_for(out_path))
    t0 = time.time()
    recs = asyncio.run(_fire(post, vmodel, system=system, user=probe["user_text"], cap=cap, S=S, evidence=evidence))
    wall = time.time() - t0
    alive = server_alive(ep)
    finish = dict(Counter(str(r.get("finish_reason")) for r in recs))
    bad = [r for r in recs if r.get("status") != 200 or r.get("finish_reason") not in OK_FINISH]
    max_prompt = max((r.get("prompt_tokens") or 0) for r in recs) if recs else 0
    ok = not bad and alive is not False and len(recs) == S
    rec = {**base, "ok": ok, "vmodel": vmodel, "max_prompt_tokens": max_prompt, "n_requests": len(recs),
           "n_ok": len(recs) - len(bad), "finish_reasons": finish, "n_bad": len(bad), "bad": bad[:20],
           "server_alive_after": alive, "wall_s": round(wall, 2),
           "request_evidence": str(evidence.path), "n_evidence_records": evidence.n, "kind_tag": f"probe/{LOAD_CHECK_KIND}",
           "verdict": ("PASS: the window carries S concurrent requests at the leg's rendered prompt + the run's cap"
                       if ok else "MEASURED LIMIT at this window: the server refused or failed a probe that FITS the "
                                  "window — re-measure the geometry for this lane through the lens standup; never shrink S silently")}
    out_path.write_text(json.dumps(rec, indent=2, default=str))
    print(f"[load-check] {'PASS' if ok else 'FAIL'}: S={S} L={L} cap={cap} ({cap_source}) prompt~{probe['rendered_prompt_tokens']} "
          f"(max served {max_prompt}) vmodel={vmodel} finish={finish} bad={len(bad)} alive={alive} wall={wall:.1f}s -> {out_path}",
          flush=True)
    return EXIT_OK if ok else EXIT_MEASURED_LIMIT


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--endpoint", default=None, help="the standup's endpoint.json (not needed with --preflight)")
    ap.add_argument("--eval", required=True, help="the geometry-table eval (families_served:short, multiturn_attacks, ...)")
    ap.add_argument("--config", default=None, help="the run's wave config: the leg's cap comes from it (never the table's)")
    ap.add_argument("--model-key", default=None, help="geometry.MODEL_KEYS key: resolves the geometry a shrunk/incompatible probe needs")
    ap.add_argument("--cap", type=int, default=None, help="override the cap (the fullconst smoke: 8192)")
    ap.add_argument("--attack", default=None, help="the multi-turn family (or comma list) this standup serves: resolves a PER-FAMILY "
                    "config cap (attacks.multiturn_target_max_new_tokens as a map) — the probe takes the max over them")
    ap.add_argument("--prompt-tokens", type=int, default=None, help="override the p99 prompt (rendered tokens)")
    ap.add_argument("--system-si", default=None, help="a registered SI id sent as the system message; the probe then hits ITS prompted route")
    ap.add_argument("--concurrency", type=int, default=None, help="S (default: the attested max_num_seqs)")
    ap.add_argument("--vmodel", default=None, help="force the vmodel (tests; the default is resolved from the routing table)")
    ap.add_argument("--preflight", action="store_true", help="validate the probe against the window with NO server (before allocating one)")
    ap.add_argument("--max-model-len", type=int, default=None, help="--preflight: the window (default: the table's via --model-key)")
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--request-evidence", default=None,
                    help="where the per-request kind-tag records go (task probe/load_check + max_tokens, the tinycap "
                         "driver's request_evidence.jsonl shape); default: <out stem>.request_evidence.jsonl beside --out. "
                         "Pass it to record_resolved_sampler.py --probe-tags")
    return ap


def main(argv=None, *, post=None) -> int:
    return run(build_parser().parse_args(argv), post=post)


if __name__ == "__main__":
    sys.exit(main())
