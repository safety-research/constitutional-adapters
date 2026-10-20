#!/usr/bin/env python
"""full22 SAMPLED re-run — the HOP-TIMEOUT ADEQUACY GUARD.

The failure shape this guard removes as a CLASS: a generation killed in transport does not look like a
failure, it looks like a SHORT completion. A ladder whose hop timeout cannot carry a rung's longest
generation therefore reads the truncated share as FALLING and exits satisfied while the long draws are
censored — it lies in the direction of "no problem here". The 600 s -> 7,200 s fix of 2026-09-10 removed
that day's instance (every cap-bound GLM draw at 8192 504'd / socket-timed-out); this guard runs BEFORE
any climb to a rung and before the GSM8K thinking-ON generation (the same failure shape), and refuses
when the binding hop cannot carry the rung.

    hop_timeout_guard.py --rung R --client static|controls --vllm-log LOG [--vllm-log LOG2 ...]
                         (--endpoint endpoint.json | --pre-standup) [--margin 2.0] [--out record.json]
                         [--repo-dir DIR] [--floor-min-running 8] [--floor-quantile 0.05] [--floor-min-samples 10]

TWO PHASES, both recorded, neither sufficient alone:
  * ``--pre-standup`` — BEFORE the rung's standup, against the value mt_serving_up.py WILL apply when the
    sbatch passes no ``--upstream-timeout-s``: ``sisteer.capeval.serving.proxy.HOP_TIMEOUT_S`` by import from
    ``--repo-dir/src``. This decides whether a standup is worth a GPU minute; it is NEVER the sole check,
    because a compiled-in default is exactly the value that censored the slow leg once.
  * ``--endpoint endpoint.json`` — AFTER the standup, from the RECORD the standup wrote, before any request.
    The sbatch runs this one unconditionally for every rung standup and refuses traffic on REFUSE.
  Passing neither refuses (undeterminable); passing both refuses (they answer different questions).

The check (one record per invocation):

1. The CONFIGURED hop timeouts, read from where the serving path records them — never a retyped number:
   * upstream (proxy -> vLLM): post-standup, ``endpoint.json`` under EITHER of
     :data:`ENDPOINT_UPSTREAM_TIMEOUT_KEYS` (the sdfbase standup mt_serving_up.py writes ``upstream_timeout_s``;
     the tinycap standup capctrl_serving_up.py writes ``proxy_upstream_timeout_s`` — two records of one fact,
     the ``cap_ladder.served_hop_timeouts`` semantics): both present and DISAGREEING is a TORN
     record and refuses; neither present refuses. Pre-standup: ``proxy.HOP_TIMEOUT_S`` (see above).
   * client (driver -> proxy): ``static`` = ``sisteer.capeval.serving.proxy.CLIENT_TIMEOUT_S`` by import (the
     ProxyClientEngine escalate_static.py builds through build_served_target); ``controls`` =
     ``scripts/served_capability_run.py:AIOHTTP_TOTAL_TIMEOUT_S`` read by AST (the controls / GSM8K driver).
     An unreadable value REFUSES: absent evidence is not evidence of adequacy.
2. ``binding = min(upstream_timeout_s, client_timeout_s)`` — the SMALLEST hop binds.
3. ``required = R / floor_tok_s x margin``. The floor decode rate is a WINDOWED floor over this model's most
   recent rung at the served geometry: over every ``Avg generation throughput: X tokens/s ... Running: N
   reqs`` line of the vLLM server log(s) named, the per-sequence rate ``X / N``; ELIGIBLE windows have
   ``N >= --floor-min-running`` (a near-empty server is not the loaded regime the rung runs in) and ``X > 0``
   (a window with no decode step carries no rate and is skipped, never a rate of zero); the floor is the
   ``--floor-quantile`` (default p05) of the eligible rates, and at least ``--floor-min-samples`` eligible
   windows are required (else undeterminable). The RAW MINIMUM over every window with N >= 1 and X > 0 is
   recorded beside it as evidence (``raw_min_tok_s`` with its window's X and N) but does not decide: a single
   prefill-heavy 10-s window (a burst of admissions, almost no decode steps) put the raw min at ~0.03 tok/s
   and the required time at ~498,000 s — a spurious refusal of a hop that carries the rung with room.
   The margin (default 2.0) is the queue wait: the drivers over-subscribe the server (~96:76), so a request
   can wait one full generation.
4. REFUSE (exit 2) when ``binding < required`` — naming the rung, the configured values and which hop binds,
   the required value, and the file / flag to change; refuse too when anything is undeterminable (no
   endpoint field, a torn record, an unreadable client constant, too few eligible windows).
5. The record ``{rung, phase, upstream_timeout_s, upstream_source, upstream_key, upstream_keys_found,
   client_timeout_s, client_source, client, binding_s, binding_hop, floor_tok_s, floor_rule, floor_quantile,
   floor_min_running, floor_min_samples, floor_n_eligible, floor_n_all, floor_samples, raw_min_tok_s,
   raw_min_running, raw_min_tps, floor_source, margin, required_s, verdict, reason, sole_check: false, tool,
   written_at}`` is written to ``--out`` and printed as ONE JSON line; ladder_extend.py carries the pre-standup
   records into decision_16384.json and REFUSES on any that is not PASS at the target rung; the sbatch folds
   the post-standup records into the ladder events, so a reader sees the check covered the run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve()
REPO_DEFAULT = HERE.parents[4]

EXIT_REFUSED = 2
DEFAULT_MARGIN = 2.0
CLIENTS = ("static", "controls")
PHASES = ("pre_standup", "post_standup")
#: the endpoint.json keys a standup records the proxy's upstream timeout under (mt_serving_up.py:
#: upstream_timeout_s; capctrl_serving_up.py: proxy_upstream_timeout_s) — cap_ladder's
#: ENDPOINT_UPSTREAM_TIMEOUT_KEYS, mirrored: accept either, refuse a disagreement
ENDPOINT_UPSTREAM_TIMEOUT_KEYS = ("upstream_timeout_s", "proxy_upstream_timeout_s")
#: the windowed floor's defaults (all three recorded on every record; CLI-overridable)
FLOOR_QUANTILE = 0.05
FLOOR_MIN_RUNNING = 8
FLOOR_MIN_SAMPLES = 10
FLOOR_RULE = ("p{q} of per-sequence decode rates (gen_tps / running) over 10-s stats windows with running >= "
              "{min_running} and gen_tps > 0, at least {min_samples} eligible windows; the raw minimum over every "
              "window with running >= 1 is recorded as evidence and does not decide")
#: what to change, per hop (the refusal names it)
FIX_FOR = {
    "upstream": "scripts/sdfbase/mt_serving_up.py --upstream-timeout-s (default sisteer.capeval.serving.proxy.HOP_TIMEOUT_S)",
    "client:static": "sisteer.capeval.serving.proxy.CLIENT_TIMEOUT_S",
    "client:controls": "scripts/served_capability_run.py AIOHTTP_TOTAL_TIMEOUT_S",
}
#: vLLM's periodic stats line (10 s cadence): generation throughput + the running count in ONE line
_STATS_RE = re.compile(r"Avg generation throughput:\s*([\d.]+)\s*tokens/s.*?Running:\s*(\d+)\s*reqs")


class Refusal(Exception):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and v > 0


def _import_proxy(repo_dir: Path):
    src = Path(repo_dir) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from sisteer.capeval.serving import proxy as pm
    except Exception as exc:  # noqa: BLE001 — a refusal, with the reason
        raise Refusal(f"cannot import sisteer.capeval.serving.proxy from {src}: {type(exc).__name__}: {exc}")
    return pm


def read_endpoint_upstream_timeout(endpoint: Path) -> Dict[str, Any]:
    """-> {seconds, source, key, keys_found} from an endpoint.json under EITHER of
    ENDPOINT_UPSTREAM_TIMEOUT_KEYS. Two keys present and disagreeing is a TORN record (refused, never a
    choice); no key, or a non-positive value, refuses (absent evidence is not evidence of adequacy)."""
    p = Path(endpoint)
    if not p.is_file():
        raise Refusal(f"--endpoint {p}: no such file (the standup has not written endpoint.json)")
    try:
        ep = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Refusal(f"--endpoint {p}: not JSON ({exc})")
    if not isinstance(ep, dict):
        raise Refusal(f"--endpoint {p}: not a JSON object")
    found = {k: ep.get(k) for k in ENDPOINT_UPSTREAM_TIMEOUT_KEYS if ep.get(k) is not None}
    bad = {k: v for k, v in found.items() if not _positive(v)}
    if bad:
        raise Refusal(f"{p}: {bad} — the endpoint's hop timeout is not a positive number (a standup older than the "
                      "7,200 s landing, or a torn record); absent evidence is not evidence of adequacy")
    if not found:
        raise Refusal(f"{p}: none of {ENDPOINT_UPSTREAM_TIMEOUT_KEYS} present — the endpoint carries no hop timeout "
                      "(a standup older than the 7,200 s landing, or a torn record); absent evidence is not evidence "
                      "of adequacy")
    if len({float(v) for v in found.values()}) > 1:
        raise Refusal(f"{p}: records the proxy upstream timeout under two keys that DISAGREE: {found} — a torn "
                      "record; fix the standup rather than pick one")
    key = next(iter(found))
    return {"seconds": float(found[key]), "source": f"{p}:{key}", "key": key, "keys_found": sorted(found)}


def read_upstream_timeout(endpoint: Optional[Path], repo_dir: Path, *, pre_standup: bool) -> Dict[str, Any]:
    """-> {seconds, source, key, keys_found, phase}. Post-standup (``endpoint``): the record. Pre-standup
    (``pre_standup``): proxy.HOP_TIMEOUT_S by import — the value the standup applies when --upstream-timeout-s
    is not passed (target_sampled.sbatch passes none). Neither, or both, refuses."""
    if endpoint is not None and pre_standup:
        raise Refusal("--endpoint and --pre-standup are mutually exclusive: the post-standup check reads the RECORD, "
                      "the pre-standup check reads the configured default; run them as two invocations")
    if endpoint is not None:
        return {**read_endpoint_upstream_timeout(endpoint), "phase": "post_standup"}
    if not pre_standup:
        raise Refusal("neither --endpoint (post-standup: the standup's endpoint.json) nor --pre-standup (the configured "
                      "default the standup will apply) given — the upstream hop timeout is undeterminable; a rung is not "
                      "climbed on absent evidence")
    pm = _import_proxy(repo_dir)
    v = getattr(pm, "HOP_TIMEOUT_S", None)
    if not _positive(v):
        raise Refusal(f"proxy.HOP_TIMEOUT_S={v!r} is not a positive number")
    return {"seconds": float(v), "key": None, "keys_found": [], "phase": "pre_standup",
            "source": "sisteer.capeval.serving.proxy.HOP_TIMEOUT_S (the standup's default; no --upstream-timeout-s passed; "
                      "PRE-standup — the post-standup --endpoint run is mandatory before any request)"}


def read_client_timeout(client: str, repo_dir: Path) -> tuple:
    """-> (seconds, source) for the driver that will issue the rung's requests."""
    repo_dir = Path(repo_dir)
    if client == "static":
        pm = _import_proxy(repo_dir)
        v = getattr(pm, "CLIENT_TIMEOUT_S", None)
        source = "sisteer.capeval.serving.proxy.CLIENT_TIMEOUT_S"
    elif client == "controls":
        path = repo_dir / "scripts" / "served_capability_run.py"
        if not path.is_file():
            raise Refusal(f"{path}: not found — cannot read AIOHTTP_TOTAL_TIMEOUT_S")
        # the constant only: parse the assignment by AST rather than importing a driver that pulls torch
        import ast
        v = None
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise Refusal(f"{path}: does not parse ({exc})")
        # the module aliases the driver imports under, for a value it DERIVES from another
        # module (main: `AIOHTTP_TOTAL_TIMEOUT_S = pe.AIOHTTP_TOTAL_TIMEOUT_S`, one number with
        # contracts.PROXY_UPSTREAM_TIMEOUT_S) — resolved by importing THAT module (light), never
        # the driver itself
        aliases = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    aliases[a.asname or a.name] = a.name
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "AIOHTTP_TOTAL_TIMEOUT_S"
                                                    for t in node.targets):
                try:
                    v = ast.literal_eval(node.value)
                except ValueError:
                    val = node.value
                    if (isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name)
                            and val.value.id in aliases):
                        import importlib
                        try:
                            mod = importlib.import_module(aliases[val.value.id])
                            v = getattr(mod, val.attr, None)
                        except Exception as exc:  # noqa: BLE001 — a refusal, never a guess
                            raise Refusal(f"{path}: AIOHTTP_TOTAL_TIMEOUT_S = {aliases[val.value.id]}."
                                          f"{val.attr} could not be resolved ({type(exc).__name__}: {exc})")
                        source_note = f" (= {aliases[val.value.id]}.{val.attr})"
                    else:
                        raise Refusal(f"{path}: AIOHTTP_TOTAL_TIMEOUT_S is not a literal nor an aliased "
                                      "module attribute")
        source = f"{path}:AIOHTTP_TOTAL_TIMEOUT_S"
    else:
        raise Refusal(f"--client {client!r} must be one of {CLIENTS}")
    if not _positive(v):
        raise Refusal(f"{source} = {v!r} is unreadable or not a positive number — refusing (absent evidence "
                      "is not evidence of adequacy)")
    return float(v), source


def floor_rate(logs: List[Path], *, min_running: int = FLOOR_MIN_RUNNING, quantile: float = FLOOR_QUANTILE,
               min_samples: int = FLOOR_MIN_SAMPLES) -> Dict[str, Any]:
    """The WINDOWED floor of the per-sequence decode rate over the named vLLM logs (FLOOR_RULE), with the raw
    minimum recorded as evidence. Refuses when fewer than ``min_samples`` eligible windows exist."""
    if min_running < 1:
        raise Refusal(f"--floor-min-running {min_running} must be >= 1")
    if not (0.0 <= quantile < 1.0):
        raise Refusal(f"--floor-quantile {quantile} must be in [0, 1)")
    if min_samples < 1:
        raise Refusal(f"--floor-min-samples {min_samples} must be >= 1")
    eligible: List[float] = []
    raw: List[tuple] = []          # (rate, tps, running) over every window with running >= 1, tps > 0
    skipped = 0
    per_log = []
    for lp in logs:
        p = Path(lp)
        if not p.is_file():
            raise Refusal(f"--vllm-log {p}: no such file — the floor decode rate is undeterminable")
        text = p.read_text(encoding="utf-8", errors="replace")
        n_here = n_elig = 0
        for m in _STATS_RE.finditer(text):
            tps, running = float(m.group(1)), int(m.group(2))
            if running < 1 or tps <= 0:
                skipped += 1
                continue
            raw.append((tps / running, tps, running))
            n_here += 1
            if running >= min_running:
                eligible.append(tps / running)
                n_elig += 1
        per_log.append({"path": str(p), "sha256": _sha(p), "n_windows": n_here, "n_eligible": n_elig})
    if len(eligible) < min_samples:
        raise Refusal(f"only {len(eligible)} eligible decode-rate window(s) (generation throughput > 0 with Running >= "
                      f"{min_running}) in {[str(l) for l in logs]}, need >= {min_samples} ({len(raw)} windows with Running "
                      f">= 1, {skipped} stats lines skipped) — the floor decode rate is undeterminable at the served load")
    eligible.sort()
    raw_min = min(raw, key=lambda t: t[0])
    return {"floor_tok_s": eligible[int(len(eligible) * quantile)],
            "floor_rule": FLOOR_RULE.format(q=int(round(quantile * 100)), min_running=min_running, min_samples=min_samples),
            "floor_quantile": float(quantile), "floor_min_running": int(min_running), "floor_min_samples": int(min_samples),
            "floor_n_eligible": len(eligible), "floor_n_all": len(raw), "floor_samples": len(eligible),
            "skipped_samples": skipped,
            "raw_min_tok_s": raw_min[0], "raw_min_tps": raw_min[1], "raw_min_running": raw_min[2],
            "eligible_min_tok_s": eligible[0], "median_tok_s": eligible[len(eligible) // 2], "max_tok_s": eligible[-1],
            "logs": per_log}


def guard(*, rung: int, client: str, logs: List[Path], endpoint: Optional[Path], pre_standup: bool, margin: float,
          repo_dir: Path, min_running: int = FLOOR_MIN_RUNNING, quantile: float = FLOOR_QUANTILE,
          min_samples: int = FLOOR_MIN_SAMPLES) -> dict:
    if isinstance(rung, bool) or rung < 1:
        raise Refusal(f"--rung {rung!r} must be a positive int")
    if margin < 1.0:
        raise Refusal(f"--margin {margin} must be >= 1.0 (the queue wait cannot be negative)")
    up = read_upstream_timeout(endpoint, repo_dir, pre_standup=pre_standup)
    cl_s, cl_src = read_client_timeout(client, repo_dir)
    fl = floor_rate(logs, min_running=min_running, quantile=quantile, min_samples=min_samples)
    up_s = up["seconds"]
    binding = min(up_s, cl_s)
    binding_hop = "upstream" if up_s <= cl_s else f"client:{client}"
    required = rung / fl["floor_tok_s"] * margin
    ok = binding >= required
    rec = {"rung": int(rung), "phase": up["phase"], "sole_check": False,
           "upstream_timeout_s": up_s, "upstream_source": up["source"], "upstream_key": up["key"],
           "upstream_keys_found": up["keys_found"],
           "client_timeout_s": cl_s, "client_source": cl_src, "client": client,
           "binding_s": binding, "binding_hop": binding_hop,
           **{k: fl[k] for k in ("floor_tok_s", "floor_rule", "floor_quantile", "floor_min_running", "floor_min_samples",
                                 "floor_n_eligible", "floor_n_all", "floor_samples", "raw_min_tok_s", "raw_min_tps",
                                 "raw_min_running", "eligible_min_tok_s")},
           "floor_source": fl["logs"], "floor_skipped_samples": fl["skipped_samples"],
           "floor_median_tok_s": fl["median_tok_s"], "floor_max_tok_s": fl["max_tok_s"],
           "margin": float(margin), "required_s": required,
           "verdict": "PASS" if ok else "REFUSE",
           "tool": {"name": HERE.name, "sha256": _sha(HERE)},
           "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    if ok:
        rec["reason"] = (f"[{up['phase']}] binding hop {binding_hop} = {binding:.0f} s carries rung {rung} at the windowed floor "
                         f"{fl['floor_tok_s']:.3f} tok/s (p{int(round(quantile * 100))} over {fl['floor_n_eligible']} windows with "
                         f"Running >= {min_running}; raw min {fl['raw_min_tok_s']:.4f} tok/s at Running {fl['raw_min_running']} "
                         f"recorded as evidence) x margin {margin} (required {required:.0f} s)")
    else:
        rec["reason"] = (f"[{up['phase']}] rung {rung}: the binding hop is {binding_hop} at {binding:.0f} s "
                         f"({'upstream' if binding_hop == 'upstream' else 'client'} timeout, {up['source'] if binding_hop == 'upstream' else cl_src}), "
                         f"but carrying {rung} tokens at the windowed floor {fl['floor_tok_s']:.3f} tok/s "
                         f"(p{int(round(quantile * 100))} over {fl['floor_n_eligible']} windows with Running >= {min_running} of "
                         f"{[l['path'] for l in fl['logs']]}; raw min {fl['raw_min_tok_s']:.4f}) x margin {margin} "
                         f"needs {required:.0f} s — a transport-killed draw would read as a SHORT completion; "
                         f"raise {FIX_FOR[binding_hop]} to >= {required:.0f} s before this rung")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--rung", type=int, required=True, help="the cap of the rung about to be generated")
    ap.add_argument("--client", required=True, choices=CLIENTS,
                    help="which driver issues the requests: static (escalate_static.py) | controls (served_capability_run.py, "
                         "the judged controls AND the GSM8K readout)")
    ap.add_argument("--vllm-log", action="append", required=True, metavar="LOG",
                    help="the vLLM server log(s) of this model's most recent rung at the served geometry (repeatable)")
    ap.add_argument("--endpoint", default=None, help="POST-standup: the rung standup's endpoint.json (the record)")
    ap.add_argument("--pre-standup", action="store_true",
                    help="PRE-standup: check the configured default the standup will apply (proxy.HOP_TIMEOUT_S by import). "
                         "Never the sole check — the --endpoint run is mandatory before any request")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--floor-min-running", type=int, default=FLOOR_MIN_RUNNING)
    ap.add_argument("--floor-quantile", type=float, default=FLOOR_QUANTILE)
    ap.add_argument("--floor-min-samples", type=int, default=FLOOR_MIN_SAMPLES)
    ap.add_argument("--out", default=None, help="write the record here (JSON)")
    ap.add_argument("--repo-dir", default=str(REPO_DEFAULT))
    args = ap.parse_args(argv)
    try:
        rec = guard(rung=args.rung, client=args.client, logs=[Path(l) for l in args.vllm_log],
                    endpoint=Path(args.endpoint) if args.endpoint else None, pre_standup=bool(args.pre_standup),
                    margin=args.margin, repo_dir=Path(args.repo_dir), min_running=args.floor_min_running,
                    quantile=args.floor_quantile, min_samples=args.floor_min_samples)
    except Refusal as exc:
        rec = {"rung": args.rung, "client": args.client, "verdict": "REFUSE", "reason": f"undeterminable: {exc}",
               "phase": "post_standup" if args.endpoint else ("pre_standup" if args.pre_standup else None),
               "sole_check": False, "tool": {"name": HERE.name, "sha256": _sha(HERE)},
               "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, sort_keys=True), flush=True)
    if rec["verdict"] != "PASS":
        print(f"[hop-timeout-guard] REFUSED: {rec['reason']}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
