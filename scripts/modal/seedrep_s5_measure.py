#!/usr/bin/env python3
"""The S5 multi-turn WINDOW MEASUREMENT producer.

Runs IN-CONTAINER (the driver venv, the staged tree), after the probe standup's drivers finished and before the log is packed, invoked by
`seedrep_served.producer_evidence` once per probe child and BOUND there like the audit (`modal_evidence.json["producer_evidence"][<rel>]`
= sha256 + writer + tree sha + rc). It reads ONLY the attempt root's own files and emits ONE `seedrep_s5_window_measurement/1` record:

    <root>/exp/transcripts_results/<private_run>/<leg>/<cell>/shard*/
        responses.jsonl + progress.jsonl   the multi-turn store (the completeness RULE, `multiturn_completeness`: COMPLETE iff every goal's
                                           reply rows equal its terminal progress record's n_target_forwards, no goal errored). The real
                                           runners persist only SAMPLER keys under progress `target_decode`; the seed base / cap / decode
                                           mode are read from the FINAL manifest's `decode` block
        target_timing.jsonl                the served target's PER-CALL record (`ServedChatTarget._one_send`, the driver process where the
                                           clamp is DECIDED): `prompt_tokens` + `completion_tokens` from the engine's `usage` on every served
                                           reply, `max_tokens_clamped_from` when the once-only clamp fired, `error: prompt_over_window` when
                                           a prompt could not fit (the multi-turn policy: a typed error, never a clamped retry)
        served_target.json                 the driver's manifest (`manifest_stage: final`): `window_clamp` counters (n_clamped,
                                           n_prompt_over_window, n_target_forwards) and the `decode` block (vendor preset, cap, seed base)
    <root>/ATTEMPT.json                    attempt_id, W (the tree sha), wave_uuid (the EXECUTION wave), launch_plan_row

The record: `context_max` = the MAXIMUM over every served call of prompt_tokens + completion_tokens (the served context; never a
percentile), `clamped_calls` = calls the client clamped (`max_tokens_clamped_from`) + calls refused as prompt_over_window, `n_calls`,
`store_complete` (store complete AND goals completed == expected), `window_source = measured_max_complete_store`, the store's identity
(leg / cell / s / arm = the STAGED id / arm_kind / object sha), the decode row ref, the cap, `produced_by {writer, tree_sha, attempt_id,
standup_id, wave_uuid}`, the store's sha256 (responses + progress + timing bytes).

WHAT IT REFUSES (rc 2, NO file -- the binding then says ABSENT and the certificate refuses that check by name): no ATTEMPT.json; no store
for (leg, cell) or more than one; no `target_timing.jsonl` (the per-call record is the ONLY source of served context lengths -- nothing is
reconstructed from the engine log); a served call without integer usage; a `served_target.json` that is not `final` or whose counters
disagree with the per-call rows; more served rows than persisted replies; the store's seed base != --s; a non-vendor decode; a resolution
record naming another policy ref / model key; more goals with progress than expected. An INCOMPLETE store or a CLAMPED call is a
MEASUREMENT (written honestly: `store_complete: false` / `clamped_calls: N`) that every consumer then refuses on content.

Argv (the app's, from the plan row -- `seedrep_launch_plan.s5_measure_argv`; nothing here is operator-typed):
    seedrep_s5_measure.py --root <attempt root> --leg crescendo --cell <defense dir> --s 1 --arm <staged id | none> --arm-kind parent|dosed
                          --staged-id <staged id | none> [--object-sha256 <sha>] --decode-policy-ref <ref> --model-key qwen35_4b
                          --n-goals-expected 259 --standup-id <row id> --out <root>/audits/s5_window/<leg>/<cell>__s<s>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
_SEEDREP = HERE.parent / "sdfbase" / "seedrep"
for _p in (HERE, _SEEDREP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402  (the store reader: the ONE completeness rule)

SCHEMA = "seedrep_s5_window_measurement/1"
WINDOW_SOURCE = "measured_max_complete_store"
WRITER = "scripts/modal/seedrep_s5_measure.py"
TIMING_FILE = "target_timing.jsonl"
MANIFEST_FILE = "served_target.json"
PROMPT_OVER_WINDOW = "prompt_over_window"
EXIT_OK, EXIT_REFUSED = 0, 2


class MeasureRefusal(RuntimeError):
    pass


def refuse(msg: str) -> None:
    raise MeasureRefusal(msg)


def _int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _sha_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def find_store(root: Path, *, leg: str, cell: str) -> List[Path]:
    """The shard dirs of the ONE multi-turn store `<private_run>/<leg>/<cell>/shard*/responses.jsonl` under the root (the audit's discovery
    shape); refuses none or several."""
    hits: Dict[Path, List[Path]] = {}
    for rp in ac.walk_files(root, "responses.jsonl"):
        defense_dir = rp.parent if not rp.parent.name.startswith("shard") else rp.parent.parent
        if defense_dir.name != cell or defense_dir.parent.name != leg:
            continue
        shards = sorted(p for p in defense_dir.glob("shard*") if p.is_dir() and (p / "responses.jsonl").is_file())
        if not shards and (defense_dir / "responses.jsonl").is_file():
            shards = [defense_dir]
        hits[defense_dir] = shards
    if not hits:
        refuse(f"{root}: no multi-turn store for ({leg}, {cell}) -- no <private_run>/{leg}/{cell}/shard*/responses.jsonl under the attempt root")
    if len(hits) > 1:
        refuse(f"{root}: {len(hits)} stores for ({leg}, {cell}) ({sorted(str(d) for d in hits)}); one probe child writes one store")
    return next(iter(hits.values()))


def read_timing(shards: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for sd in shards:
        tp = sd / TIMING_FILE
        if not tp.is_file():
            refuse(f"{sd}: no {TIMING_FILE} -- the served target's per-call record (prompt_tokens + completion_tokens, the clamp marker) is the ONLY "
                   "source of served context lengths; nothing is reconstructed from the engine log")
        rows.extend(ac.read_jsonl(tp))
    return rows


def multiturn_completeness(rows: Sequence[dict], progress: Sequence[dict]) -> Dict[str, Any]:
    """The multi-turn completeness rule (`audit_common.read_multiturn_store`), applied to the REAL writer's rows (the real runners'
    progress `target_decode` carries sampler keys only -- no seed_base / cap -- so that reader, which int()s `target_decode.seed_base`,
    cannot be called on them; the rule itself is unchanged): one terminal progress record per goal, its outcome not an error, its
    `n_target_forwards` (one ATTEMPTED target forward per reply; `ServedChatTarget.chat` counts before sending) == the reply rows stored
    for the goal, each (goal, order) once."""
    for r in rows:
        if "order" not in r or "goal_id" not in r:
            refuse("a responses.jsonl row lacks goal_id / order -- not a multiturn.run row")
    incomplete: List[str] = []
    n_declared: Dict[str, int] = {}
    for p_ in progress:
        g = str(p_.get("goal_id"))
        if str(p_.get("outcome", "")).startswith("error"):
            incomplete.append(f"goal {g!r} finalized with outcome {p_.get('outcome')!r}")
        n = p_.get("n_target_forwards")
        if isinstance(n, bool) or not isinstance(n, int):
            incomplete.append(f"goal {g!r}: the progress record carries no integer n_target_forwards ({n!r})")
            continue
        if g in n_declared:
            incomplete.append(f"goal {g!r} has two terminal progress records")
        n_declared[g] = n
    stored: Dict[str, int] = {}
    seen = set()
    dup = []
    for r in rows:
        key = (str(r["goal_id"]), int(r["order"]))
        if key in seen:
            dup.append(key)
        seen.add(key)
        stored[key[0]] = stored.get(key[0], 0) + 1
    for g, n in sorted(n_declared.items()):
        if stored.get(g, 0) != n:
            incomplete.append(f"goal {g!r}: {stored.get(g, 0)} stored reply row(s) != n_target_forwards {n} the progress record declares")
    extra = sorted(set(stored) - set(n_declared))
    if extra:
        incomplete.append(f"{len(extra)} goal(s) with reply rows but no progress record (first: {extra[:3]})")
    if dup:
        incomplete.append(f"{len(dup)} (goal_id, order) stored twice (first: {dup[:3]})")
    if not n_declared:
        incomplete.append("no progress record declares a goal -- an empty store")
    return {"complete": not incomplete, "n_rows": len(rows), "goals": set(n_declared),
            "note": ("every goal's reply rows equal its progress record's n_target_forwards, each (goal_id, order) once" if not incomplete else " | ".join(incomplete))}


def read_manifest(shards: Sequence[Path]) -> dict:
    mans = [sd / MANIFEST_FILE for sd in shards if (sd / MANIFEST_FILE).is_file()]
    if not mans:
        refuse(f"{shards[0].parent}: no {MANIFEST_FILE} beside the store -- the driver never wrote the served target's manifest")
    if len(mans) > 1:
        refuse(f"{shards[0].parent}: {len(mans)} {MANIFEST_FILE} files (one store, one manifest)")
    m = ac.read_json(mans[0])
    if not isinstance(m, dict):
        refuse(f"{mans[0]}: not an object")
    if m.get("manifest_stage") != "final":
        refuse(f"{mans[0]}: manifest_stage {m.get('manifest_stage')!r} is not 'final' -- the driver did not finish (its clamp counters are not real)")
    return m


def measure(*, root: Path, leg: str, cell: str, s: int, arm: str, arm_kind: str, staged_id: str, object_sha256: Optional[str], decode_policy_ref: str,
            model_key: str, n_goals_expected: int, standup_id: str, measured_utc: Optional[str] = None) -> Dict[str, Any]:
    root = Path(root)
    ap = root / ac.ATTEMPT_FILE
    if not ap.is_file():
        refuse(f"{root}: no {ac.ATTEMPT_FILE} -- not an attempt root")
    att = ac.read_json(ap)
    if not isinstance(att, dict) or not isinstance(att.get("attempt_id"), str) or not isinstance(att.get("W"), str) or not isinstance(att.get("wave_uuid"), str):
        refuse(f"{ap}: attempt_id / W / wave_uuid missing -- the measurement cannot bind to an attempt")
    if att.get("launch_plan_row") not in (None, standup_id):
        refuse(f"{ap}: launch_plan_row {att.get('launch_plan_row')!r} is not the row this measurement is for ({standup_id!r})")
    if arm_kind not in ("parent", "dosed"):
        refuse(f"--arm-kind {arm_kind!r} must be parent | dosed")
    if arm_kind == "dosed" and not object_sha256:
        refuse("a dosed arm names the object it measured: --object-sha256 is required")
    shards = find_store(root, leg=leg, cell=cell)
    store_dir = shards[0].parent if shards[0].name.startswith("shard") else shards[0]
    rows: List[dict] = []
    progress: List[dict] = []
    for sd in shards:
        rows.extend(ac.read_jsonl(sd / "responses.jsonl"))
        pp = sd / "progress.jsonl"
        if not pp.is_file():
            refuse(f"{sd}: responses.jsonl without progress.jsonl -- a multi-turn store is the pair")
        progress.extend(ac.read_jsonl(pp))
    if not rows:
        refuse(f"{store_dir}: no multi-turn rows")
    sr = multiturn_completeness(rows, progress)
    # the seed base, the cap and the decode MODE are the driver's FINAL manifest's `decode` block (jailbreak_run's mt_decode_block: the
    # identity the store was drawn under); the runners persist only the SAMPLER keys under progress `target_decode` -- read the identity from the
    # manifest, hold the progress rows' sampler keys to it where both name a key
    man = read_manifest(shards)
    dec = man.get("decode") if isinstance(man.get("decode"), dict) else None
    if not dec:
        refuse(f"{store_dir}/{MANIFEST_FILE}: no `decode` block -- the store does not say which target decode / cap / seed base it was drawn under")
    if dec.get("multiturn_target_decode") != "vendor":
        refuse(f"{store_dir}/{MANIFEST_FILE}: decode.multiturn_target_decode {dec.get('multiturn_target_decode')!r} is not the vendor preset -- not the battery's decode row")
    cap = _int(dec.get("cap"))
    if cap is None or cap <= 0:
        refuse(f"{store_dir}/{MANIFEST_FILE}: decode.cap {dec.get('cap')!r} is not the served per-reply cap")
    if _int(dec.get("seed_base")) is None or int(dec["seed_base"]) != int(s):
        refuse(f"{store_dir}/{MANIFEST_FILE}: decode.seed_base {dec.get('seed_base')!r} is not the child's --s {s}")
    tds = [p_.get("target_decode") for p_ in progress if isinstance(p_.get("target_decode"), dict)]
    for t in tds:
        dis = {k: (t[k], dec.get(k)) for k in t if k in dec and t[k] != dec[k]}
        if dis:
            refuse(f"{store_dir}: a progress row's target_decode disagrees with the manifest's decode block on {dis}")
    windows = {_int(p_.get("served_window")) for p_ in progress}
    served_window = next(iter(windows)) if len(windows) == 1 else None
    if len(windows) > 1:
        refuse(f"{store_dir}: progress rows record served_window {sorted(str(w) for w in windows)}; one standup serves one window")
    res = man.get("target_decode_resolution") if isinstance(man.get("target_decode_resolution"), dict) else {}
    ref_rec = res.get("ref") or res.get("policy_ref")
    if ref_rec is not None and str(ref_rec) != str(decode_policy_ref):
        refuse(f"{store_dir}/{MANIFEST_FILE}: the driver resolved decode row {ref_rec!r}, not the row this child was invoked with ({decode_policy_ref!r})")
    mk_rec = res.get("model_key")
    if mk_rec is not None and str(mk_rec) != str(model_key):
        refuse(f"{store_dir}/{MANIFEST_FILE}: the driver resolved model key {mk_rec!r}, not {model_key!r}")
    if isinstance(man.get("cell_id"), str) and man["cell_id"] != cell:
        refuse(f"{store_dir}/{MANIFEST_FILE}: cell_id {man['cell_id']!r} is not the store's defense dir {cell!r}")
    # --- the per-call record: every served reply's context, every clamp / over-window call ------------------------------------------
    timing = read_timing(shards)
    served = [r for r in timing if r.get("error") is None]
    over = [r for r in timing if r.get("error") == PROMPT_OVER_WINDOW]
    other_err = [r for r in timing if r.get("error") not in (None, PROMPT_OVER_WINDOW)]
    clamped_retries = [r for r in timing if r.get("max_tokens_clamped_from") is not None]
    ctx: List[int] = []
    ctx_call: Optional[Dict[str, Any]] = None
    for i, r in enumerate(served):
        pt, ct = _int(r.get("prompt_tokens")), _int(r.get("completion_tokens"))
        if pt is None or ct is None or pt <= 0 or ct < 0:
            refuse(f"{store_dir}/{TIMING_FILE}: served call {i} carries no integer usage (prompt_tokens {r.get('prompt_tokens')!r}, completion_tokens "
                   f"{r.get('completion_tokens')!r}) -- the engine's usage is the served context; nothing is guessed")
        ctx.append(pt + ct)
        if ctx_call is None or pt + ct > ctx_call["context"]:
            ctx_call = {"index": i, "prompt_tokens": pt, "completion_tokens": ct, "context": pt + ct, "max_tokens": _int(r.get("max_tokens")),
                        "finish_reason": r.get("finish_reason"), "t_start": r.get("t_start")}
    if not ctx:
        refuse(f"{store_dir}/{TIMING_FILE}: no served call -- nothing to measure")
    if len(served) != sr["n_rows"]:
        refuse(f"{store_dir}: {len(served)} served call(s) in {TIMING_FILE} but {sr['n_rows']} persisted reply row(s) -- the per-call record and the store disagree")
    # `ServedChatTarget.chat` counts a forward BEFORE sending, so n_target_forwards = served + over-window + other failed calls -- every
    # attempted call must be accounted for by a timing row (served or failed); only an UNACCOUNTED call refuses
    wc = man.get("window_clamp") if isinstance(man.get("window_clamp"), dict) else {}
    checks = {"n_clamped": (_int(wc.get("n_clamped")), len(clamped_retries)), "n_prompt_over_window": (_int(wc.get("n_prompt_over_window")), len(over)),
              "n_target_forwards": (_int(wc.get("n_target_forwards")), len(timing))}
    bad = {k: v for k, v in checks.items() if v[0] is None or v[0] != v[1]}
    if bad:
        refuse(f"{store_dir}/{MANIFEST_FILE}: window_clamp counters disagree with the per-call rows: " + ", ".join(f"{k} manifest {v[0]!r} vs rows {v[1]}" for k, v in bad.items()))
    if served_window is not None and max(ctx) > int(served_window):
        refuse(f"{store_dir}: a served call's context {max(ctx)} exceeds the served window {served_window} the progress rows record -- the per-call record is not this store's")
    # --- goals: the writer's terminal records vs the plan's expectation -------------------------------------------------------------
    goals_prog = {str(p["goal_id"]) for p in progress if "goal_id" in p}
    errored = sorted(str(p["goal_id"]) for p in progress if str(p.get("outcome", "")).startswith("error"))
    over_goals = [g for g in errored if "PromptOverWindow" in str(next(p.get("outcome") for p in progress if str(p.get("goal_id")) == g))]
    completed = len(goals_prog) - len(set(errored))
    if len(goals_prog) > int(n_goals_expected):
        refuse(f"{store_dir}: {len(goals_prog)} goals carry a progress record but the plan expects {n_goals_expected} for this arm-run / shard")
    if len(over_goals) > len(over):
        refuse(f"{store_dir}: {len(over_goals)} goal(s) ended error:PromptOverWindow but {TIMING_FILE} records {len(over)} prompt_over_window call(s)")
    if other_err:
        refuse(f"{store_dir}/{TIMING_FILE}: {len(other_err)} call(s) failed with another error ({sorted({str(r.get('error')) for r in other_err})[:3]}) -- the store is not a clean measurement")
    store_complete = bool(sr["complete"]) and completed == int(n_goals_expected) and int(n_goals_expected) > 0
    clamped_calls = len(clamped_retries) + len(over)
    blobs = []
    for sd in shards:
        for name in ("responses.jsonl", "progress.jsonl"):
            blobs.append((sd / name).read_bytes())
    timing_sha = _sha_bytes(*[(sd / TIMING_FILE).read_bytes() for sd in shards if (sd / TIMING_FILE).is_file()])
    try:
        store_rel = str(store_dir.relative_to(root))
    except ValueError:
        store_rel = str(store_dir)
    rec: Dict[str, Any] = {
        "schema": SCHEMA, "window_source": WINDOW_SOURCE, "leg": leg, "arm": arm, "arm_kind": arm_kind, "model_key": model_key,
        "decode_policy_ref": decode_policy_ref, "objects_measured": [staged_id], "store_path": store_rel,
        "n_goals_completed": int(completed), "n_goals_expected": int(n_goals_expected), "store_complete": store_complete,
        "context_max": int(max(ctx)), "clamped_calls": int(clamped_calls), "cap": cap,
        "measured_utc": measured_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wave_uuid": att["wave_uuid"], "attempt_id": att["attempt_id"], "decode": "vendor",
        # W16s5 extras (S5_WINDOW_MEASUREMENT_EXTRA_KEYS): the store's identity, the counts behind the two numbers, the provenance
        "cell": cell, "s": int(s), "staged_id": staged_id, "object_sha256": object_sha256, "n_calls": len(timing),
        "n_prompt_over_window": len(over), "n_clamped_retries": len(clamped_retries), "complete": bool(sr["complete"]), "completeness_note": sr["note"],
        "served_window": served_window, "context_max_call": ctx_call, "store_sha256": _sha_bytes(*blobs), "timing_sha256": timing_sha,
        "n_goals_with_progress": len(goals_prog), "n_goals_errored": len(set(errored)),
        "produced_by": {"writer": WRITER, "tree_sha": att["W"], "attempt_id": att["attempt_id"], "standup_id": standup_id, "wave_uuid": att["wave_uuid"]},
    }
    return rec


def write_record(rec: Mapping[str, Any], out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(rec), indent=1, sort_keys=False) + "\n")
    os.replace(tmp, out)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the attempt root (ATTEMPT.json + exp/ as the drivers wrote it)")
    ap.add_argument("--leg", required=True, choices=list(ac.MULTITURN_ATTACKS))
    ap.add_argument("--cell", required=True, help="the store's defense dir (the driver's --defense)")
    ap.add_argument("--s", type=int, required=True, help="the child's seed base (the store's target_decode.seed_base must equal it)")
    ap.add_argument("--arm", required=True, help="the measurement's arm: the STAGED id of the probe cell, or `none`")
    ap.add_argument("--arm-kind", required=True, choices=["parent", "dosed"])
    ap.add_argument("--staged-id", required=True)
    ap.add_argument("--object-sha256", default=None)
    ap.add_argument("--decode-policy-ref", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--n-goals-expected", type=int, required=True, help="the plan's goals per arm-run for this shard (completeness is measured against it)")
    ap.add_argument("--standup-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        rec = measure(root=a.root, leg=a.leg, cell=a.cell, s=a.s, arm=a.arm, arm_kind=a.arm_kind, staged_id=a.staged_id, object_sha256=a.object_sha256,
                      decode_policy_ref=a.decode_policy_ref, model_key=a.model_key, n_goals_expected=a.n_goals_expected, standup_id=a.standup_id)
    except (MeasureRefusal, ac.AuditRefusal) as e:
        print(f"[s5-measure] REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    out = write_record(rec, a.out)
    print(f"[s5-measure] {rec['leg']}/{rec['arm']} ({rec['arm_kind']}, cell {rec['cell']} s{rec['s']}): context_max={rec['context_max']} clamped_calls={rec['clamped_calls']} "
          f"n_calls={rec['n_calls']} goals {rec['n_goals_completed']}/{rec['n_goals_expected']} store_complete={rec['store_complete']} -> {out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
