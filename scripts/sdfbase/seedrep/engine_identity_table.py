#!/usr/bin/env python3
"""The Qwen seed-replicate battery — the engine-identity table.

    engine_identity_table.py --stores RS/stores --out RS/tables/engine_identity.csv --legs-out RS/tables/engine_identity_legs.json
        --arditi-pins configs/arditi_allpos_pins.json [--arditi-pin-key qwen35_4b/ar_c0p3__arditi_allpos] [--W <sha>]

One row per STANDUP — every pulled accepted attempt root (view engines ×3, the Arditi engine, every rung, the Petri
proxies), from its `endpoint.json` / `serving_attestation.json` / `standup_jobs.jsonl`: W (`standup_jobs.jsonl[0].sha`,
the tree the standup served from — held equal to `ATTEMPT.json.W` and to `--W`), model revision, template kwargs
(`default_chat_template_kwargs`), `server_default_policy`, S × L (`attested_max_num_seqs` × `attested_max_model_len`),
`attested_cudagraph_mode` + the FULL capture max (held ≥ S), `lora_config.max_lora_rank` / `max_loras`,
`serving_config_sha256` (the endpoint's key, else the sha256 of the pulled `config_used` file), the adapter
fingerprints (`serving_attestation.adapters[*].fingerprint_bf16` [+ `residual_fingerprint`]), the kernel-ceiling step
(`geometry.adjustments` kind `lora_kernel_index_ceiling`: from_S → to_S), and the Arditi pin verdict (the endpoint's
`object_artifact_sha256_by_cell[arditi_allpos]` AND `expected_object_sha256_by_cell[arditi_allpos]` equal the pins
file's `object_sha256`, and the attestation names a `vector_all` adapter).

Per CONTAINER leg the table then ASSERTS: all four base engines present (s1, s2, s3, arditi) and the five
identity fields — W, model revision, template kwargs, server_default_policy, S × L — equal across them; every rung
engine of the leg equal on the first four and equal on S × L within its (leg, cap); every engine's capture max ≥ its S;
the Arditi engine's pin verdict `pin_ok`. CUDA-graph capture sizes (`composed_cudagraph_capture_sizes`, the list the
composer handed `--cudagraph-capture-sizes`) are NOT an identity axis of their own but a FUNCTION of the attested S and
the FULL-capture rule — `scripts/tinycap_math500_thinking.py` `CAPTURE_SIZES_RULE` /
`capture_sizes_by_derivation`: two legs on the same window class with the same attested S must carry IDENTICAL lists
(a difference is a problem), and a leg with another attested S compares its list against the composer's derivation
for ITS OWN S (`vllm_server.cudagraph_capture_sizes_for(S)`), never against the other's. A leg that fails is written to `engine_identity_legs.json` as FAIL with the
named problems (coverage reads it as the miss "Arditi absent/mismatched" / "identity: …"). A row whose identity field
is unreadable REFUSES the table (exit 2) — no identity column is ever blank. Exit 1 when the table is written and a
leg FAILS.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_common as ac  # noqa: E402
from audit_common import AttemptRoot, Standup, refuse  # noqa: E402

COLUMNS: Tuple[str, ...] = (
    "leg", "view", "stage", "shard", "attempt_id", "W", "model_revision", "template_kwargs", "server_default_policy",
    "S", "L", "SxL", "cudagraph_mode", "capture_max", "capture_ok", "capture_sizes", "capture_sizes_derived_ok", "max_lora_rank",
    "max_loras", "serving_config_sha256", "adapter_fingerprints", "n_adapters", "kernel_from_S", "kernel_to_S", "carried_from_W",
    "arditi_pin_verdict", "engine", "thinking_mode", "max_logprobs", "problems", "endpoint_path",
    # the standup record's provenance (F-AA): `standup_jobs.jsonl` when the standup wrote one, else the derivation named
    "standup_record", "W_source")
IDENTITY_FIELDS: Tuple[str, ...] = ("W", "model_revision", "template_kwargs", "server_default_policy", "SxL")
#: the `carried_from_W` cell of an engine the carry mapping does NOT name -- an explicit value, never a null (`write_csv` refuses one)
NOT_CARRIED = "not_carried"


def _is_carried(row: Dict[str, Any]) -> bool:
    return bool(row.get("carried_from_W")) and row.get("carried_from_W") != NOT_CARRIED
#: the endpoint key `native_standup` records the composed `--cudagraph-capture-sizes` list under
CAPTURE_SIZES_KEY = "composed_cudagraph_capture_sizes"
#: the rule this table applies to that list (origin/main b231fc7e; `scripts/tinycap_math500_thinking.py`
#: `CAPTURE_SIZES_RULE` + `capture_sizes_by_derivation`): the list is `vllm_server.cudagraph_capture_sizes_for(S)` — vLLM's
#: default capture sizes for S plus S itself, sorted — a FUNCTION of the attested S and the FULL-capture rule (#162), so a
#: difference between same-engine legs is admitted ONLY as each side's derivation for its own attested S
CAPTURE_SIZES_RULE = ("main b231fc7e (PR #210) tinycap_math500_thinking.CAPTURE_SIZES_RULE: capture sizes = "
                      "vllm_server.cudagraph_capture_sizes_for(attested S) — vLLM's default sizes for S plus S itself, sorted; a "
                      "function of the attested S and the FULL-capture rule (#162), not an identity axis")


def capture_sizes_derivation(S: int) -> List[int]:
    """The composer's OWN list for width S (`vllm_server.cudagraph_capture_sizes_for`, torch-free) — derived, never trusted."""
    from sisteer.capeval.serving import vllm_server as VS
    return sorted(int(x) for x in VS.cudagraph_capture_sizes_for(int(S)))


def capture_sizes_of(standup: Standup) -> List[int]:
    v = standup.endpoint.get(CAPTURE_SIZES_KEY)
    if not isinstance(v, list) or not v or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        refuse(f"{standup.endpoint_path}: {CAPTURE_SIZES_KEY} missing or not a non-empty list of ints ({v!r}) — the composed "
               "capture list is an identity input (its derivation from the attested S is asserted); a blank is not written")
    return sorted(int(x) for x in v)


PIN_OK = "pin_ok"
NOT_ARDITI = "not_arditi"
DEFAULT_PIN_KEY = f"{ac.MODEL_KEY}/{ac.ARDITI_STAGED_ID}"


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def _required(ep: Dict[str, Any], key: str, where: Path) -> Any:
    if key not in ep or ep[key] is None:
        refuse(f"{where}: identity field {key!r} missing — an engine row with a blank identity column is not written")
    return ep[key]


def serving_config_sha256(standup: Standup) -> str:
    ep = standup.endpoint
    v = ep.get("serving_config_sha256")
    if isinstance(v, str) and v:
        return v
    cfg = ep.get("config_used")
    if isinstance(cfg, str) and cfg:
        hits = ac.walk_files(standup.root.path, Path(cfg).name)
        if len(hits) == 1:
            return ac.sha256_file(hits[0])
        if len(hits) > 1:
            refuse(f"{standup.endpoint_path}: no serving_config_sha256 and {len(hits)} copies of config_used {Path(cfg).name!r} under the root")
    refuse(f"{standup.endpoint_path}: no serving_config_sha256 and the pulled root holds no copy of config_used "
           f"({cfg!r}) — the serving config identity cannot be established")


def adapter_fingerprints(standup: Standup) -> Dict[str, Dict[str, Optional[str]]]:
    adapters = standup.attestation.get("adapters")
    if not isinstance(adapters, dict) or not adapters:
        refuse(f"{standup.attestation_path}: no adapters block")
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for name, blk in sorted(adapters.items()):
        if not isinstance(blk, dict) or not blk.get("fingerprint_bf16"):
            refuse(f"{standup.attestation_path}: adapter {name!r} carries no fingerprint_bf16")
        out[str(name)] = {"fingerprint_bf16": str(blk["fingerprint_bf16"]),
                          "residual_fingerprint": blk.get("residual_fingerprint"),
                          "petri_source_kind": blk.get("petri_source_kind")}
    return out


def arditi_pin_verdict(standup: Standup, *, pins: Optional[Dict[str, Any]], pin_key: str) -> str:
    if standup.root.view != ac.ARDITI_VIEW:
        return NOT_ARDITI
    if pins is None:
        refuse("an Arditi standup is present but --arditi-pins was not given — its pin cannot be verified")
    model_key, _, staged = pin_key.partition("/")
    entry = (pins.get(model_key) or {}).get(staged) if isinstance(pins, dict) else None
    if not isinstance(entry, dict) or not entry.get("object_sha256"):
        refuse(f"pins file has no {pin_key!r} entry with object_sha256")
    want = str(entry["object_sha256"])
    ep = standup.endpoint
    got = (ep.get("object_artifact_sha256_by_cell") or {}).get(ac.ARDITI_CELL)
    exp = (ep.get("expected_object_sha256_by_cell") or {}).get(ac.ARDITI_CELL)
    problems = []
    if got != want:
        problems.append(f"object_artifact_sha256_by_cell[{ac.ARDITI_CELL}]={got!r} != pin {want[:12]}…")
    if exp != want:
        problems.append(f"expected_object_sha256_by_cell[{ac.ARDITI_CELL}]={exp!r} != pin {want[:12]}…")
    kinds = {blk.get("petri_source_kind") for blk in (standup.attestation.get("adapters") or {}).values() if isinstance(blk, dict)}
    if entry.get("kind", "vector_all") not in kinds:
        problems.append(f"attestation names no {entry.get('kind', 'vector_all')!r} adapter (kinds: {sorted(str(k) for k in kinds)})")
    return PIN_OK if not problems else "pin_mismatch: " + "; ".join(problems)


def engine_row(standup: Standup, *, pins: Optional[Dict[str, Any]], pin_key: str, W_expected: Optional[str],
               carried_W: Optional[str] = None) -> Dict[str, Any]:
    """``carried_W``: the source tree's W a CARRIED attempt is held to instead of ``W_expected`` -- its engine identity was established
    against its own tree's pin by the carry record's engine-identity clause; the row records it under `carried_from_W`."""
    root = standup.root
    ep = standup.endpoint
    where = standup.endpoint_path
    W = standup.W  # standup_jobs.jsonl[0].sha (a line without one refuses), else ATTEMPT.json's W with the derivation recorded
    problems: List[str] = []
    if standup.standup_jobs and W != root.W:
        problems.append(f"standup_jobs.jsonl sha {W} != ATTEMPT.json W {root.W}")
    if carried_W:
        if W != carried_W:
            problems.append(f"carried standup sha {W} != its carry source W {carried_W}")
    elif W_expected and W != W_expected:
        problems.append(f"standup sha {W} != --W {W_expected}")
    S, L = standup.S, standup.L
    mode = _required(ep, "attested_cudagraph_mode", where)
    cap_max = _required(ep, "attested_cudagraph_full_capture_max", where)
    lora = _required(ep, "lora_config", where)
    if not isinstance(lora, dict) or lora.get("max_lora_rank") is None or lora.get("max_loras") is None:
        refuse(f"{where}: lora_config lacks max_lora_rank / max_loras ({lora!r})")
    kern = ac.kernel_ceiling_adjustment(ep)
    if kern is None:
        refuse(f"{where}: geometry.adjustments records no lora_kernel_index_ceiling step — the served S is not the "
               "kernel-lowered S the plan asserts (§4.3), or the record is incomplete")
    if int(kern["to_S"]) != S:
        problems.append(f"kernel ceiling to_S {kern['to_S']} != attested S {S}")
    capture_ok = int(cap_max) >= S
    if not capture_ok:
        problems.append(f"FULL capture max {cap_max} < S {S}")
    if str(mode).upper() != "FULL":
        problems.append(f"attested_cudagraph_mode {mode!r} is not FULL")
    sizes = capture_sizes_of(standup)
    want = capture_sizes_derivation(S)
    sizes_ok = sizes == want
    if not sizes_ok:
        problems.append(f"{CAPTURE_SIZES_KEY} {sizes} is not the composer's derivation for its own attested S {S} ({want}; "
                        f"{CAPTURE_SIZES_RULE}) — a real setting change, not the width")
    fps = adapter_fingerprints(standup)
    pin = arditi_pin_verdict(standup, pins=pins, pin_key=pin_key)
    if pin not in (PIN_OK, NOT_ARDITI):
        problems.append(pin)
    tk = _required(ep, "default_chat_template_kwargs", where) if "default_chat_template_kwargs" in ep else None
    if "default_chat_template_kwargs" not in ep:
        refuse(f"{where}: default_chat_template_kwargs missing (an identity field; null is a legal VALUE but the key must be present)")
    if "server_default_policy" not in ep:
        refuse(f"{where}: server_default_policy missing")
    return {"leg": root.leg, "view": root.view, "stage": root.stage, "shard": root.shard, "attempt_id": root.attempt_id,
            "W": W, "model_revision": str(_required(ep, "model_revision", where)), "template_kwargs": _canon(tk),
            "server_default_policy": _canon(ep["server_default_policy"]), "S": S, "L": L, "SxL": f"{S}x{L}",
            "cudagraph_mode": str(mode), "capture_max": int(cap_max), "capture_ok": capture_ok,
            "capture_sizes": sizes, "capture_sizes_derived_ok": sizes_ok,
            "max_lora_rank": int(lora["max_lora_rank"]), "max_loras": int(lora["max_loras"]),
            "serving_config_sha256": serving_config_sha256(standup), "adapter_fingerprints": fps, "n_adapters": len(fps),
            "kernel_from_S": int(kern["from_S"]), "kernel_to_S": int(kern["to_S"]), "arditi_pin_verdict": pin,
            "engine": str(_required(ep, "engine", where)), "thinking_mode": _canon(ep.get("thinking_mode")),
            "max_logprobs": _canon(ep.get("max_logprobs")), "problems": problems, "endpoint_path": str(where), "carried_from_W": carried_W or NOT_CARRIED,
            "standup_record": standup.record_source, "W_source": str(standup.provenance.get("W_source"))}


def leg_verdicts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per container leg: the four base engines present and identical on the five fields; rungs identical on the
    first four and on S × L within (leg, stage); the composed capture list identical across engines of one attested S
    (a derivation of S, never an identity axis of its own); no row problems.

    The cross-engine requirement on `W` is engine version / config identity, not tree identity, for a VALIDATED carry -- a row
    with `carried_from_W` (held to its OWN source tree's W in :func:`engine_row`) is exempt from the leg's W agreement, while every row the
    carry mapping does not name must still share ONE W (a differing W there fails as before). Every other identity field is held across
    ALL engines, carried or not."""
    by_leg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_leg[r["leg"]].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for leg, rs in sorted(by_leg.items()):
        problems: List[str] = []
        base = [r for r in rs if r["stage"] == ac.STAGE_BASE]
        views_present = sorted({r["view"] for r in base})
        missing = [v for v in (*ac.SEED_VIEWS, ac.ARDITI_VIEW) if v not in views_present]
        if missing:
            problems.append(("Arditi absent" if missing == [ac.ARDITI_VIEW] else f"base engine(s) absent for view(s) {missing}"))
        fresh = [r for r in rs if not _is_carried(r)]
        for f in IDENTITY_FIELDS:
            pool = [r for r in base if f != "W" or not _is_carried(r)]       # W agreement over the engines the carry does not name
            vals = {r[f] for r in pool}
            if len(vals) > 1:
                problems.append(f"{f} differs across base engines" + (" not named by the carry mapping" if f == "W" and len(pool) != len(base) else "") + f": {sorted(vals)}")
        if base:
            ref = base[0]
            # ONE W across ALL uncarried rows, whatever their stage -- the reference is the first uncarried row of
            # any kind (a base when one is fresh, else the first fresh rung), so a leg whose every base is carried still holds its fresh
            # rungs to one tree; a carried row (base or rung) is held to its own source W in engine_row instead
            uncarried = [r for r in fresh if r["stage"] == ac.STAGE_BASE] + [r for r in fresh if r["stage"] != ac.STAGE_BASE]
            w_ref = uncarried[0]["W"] if uncarried else None
            for r in rs:
                if r["stage"] == ac.STAGE_BASE:
                    continue
                for f in IDENTITY_FIELDS[:-1]:
                    if f == "W":
                        if _is_carried(r):
                            continue
                        if w_ref is not None and r[f] != w_ref:
                            problems.append(f"rung {r['view']}/{r['stage']}/{r['shard']}: W {r[f]!r} != the uncarried engines' {w_ref!r}")
                        continue
                    if r[f] != ref[f]:
                        problems.append(f"rung {r['view']}/{r['stage']}/{r['shard']}: {f} {r[f]!r} != base {ref[f]!r}")
        by_stage: Dict[str, set] = defaultdict(set)
        for r in rs:
            by_stage[r["stage"]].add(r["SxL"])
        for stage, sxl in sorted(by_stage.items()):
            if len(sxl) > 1:
                problems.append(f"stage {stage}: S×L differs across its engines: {sorted(sxl)}")
        # capture sizes: a FUNCTION of the attested S. Redundant BY CONSTRUCTION with engine_row's per-row
        # derivation check (rows equal to f(S) are identical for one S) — kept as the REPORTING line that names the engines
        # of one attested S whose lists disagree; engines of different S rightly differ
        by_S: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for r in rs:
            by_S[int(r["S"])][json.dumps(r["capture_sizes"])].append(f"{r['view']}/{r['stage']}/{r['shard']}")
        for S, lists in sorted(by_S.items()):
            if len(lists) > 1:
                problems.append(f"attested S {S}: {CAPTURE_SIZES_KEY} differs across engines of one attested S: "
                                + "; ".join(f"{k} on {sorted(v)}" for k, v in sorted(lists.items())) + f" ({CAPTURE_SIZES_RULE})")
        for r in rs:
            for p in r["problems"]:
                problems.append(f"{r['view']}/{r['stage']}/{r['shard']}: {p}")
        carried_ws = sorted({r["carried_from_W"] for r in base if _is_carried(r)})
        fresh_base = [r for r in base if not _is_carried(r)]
        identity = None
        if base and not any("differs across base" in p for p in problems):
            identity = {f: base[0][f] for f in IDENTITY_FIELDS}
            identity["W"] = fresh_base[0]["W"] if fresh_base else base[0]["W"]        # The wave's W is the fresh engines'; a fully carried leg keeps the source's
            if carried_ws:
                identity["carried_from_W"] = carried_ws
        out[leg] = {"verdict": "PASS" if not problems else "FAIL", "problems": problems,
                    "n_engines": len(rs), "base_views": views_present, "n_carried": sum(1 for r in base if _is_carried(r)),
                    "identity": identity, "stages": {st: sorted(v) for st, v in by_stage.items()}}
    return out


def build(roots: Sequence[AttemptRoot], *, pins: Optional[Dict[str, Any]], pin_key: str, W_expected: Optional[str],
          carried: Optional[Mapping[str, str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """``carried``: `{attempt_id: source W}` from the VALIDATED carry record (`audit_common.carry_source_ws`); a root it names is held to
    that W (never to ``W_expected``) and exempt from the leg's cross-engine W agreement; a root it does not name behaves exactly as before."""
    rows = [engine_row(ac.read_standup(r), pins=pins, pin_key=pin_key, W_expected=W_expected, carried_W=(carried or {}).get(r.attempt_id)) for r in roots]
    return rows, leg_verdicts(rows)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stores", type=Path, default=None)
    ap.add_argument("--roots", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--legs-out", type=Path, default=None, help="default: <out dir>/engine_identity_legs.json")
    ap.add_argument("--arditi-pins", type=Path, default=None, help="configs/arditi_allpos_pins.json")
    ap.add_argument("--arditi-pin-key", default=DEFAULT_PIN_KEY)
    ap.add_argument("--W", default=None, help="the cut W every standup must have served from")
    ap.add_argument("--carry-over", type=Path, default=None, help="RS/carry_over.json (D125): a CARRIED standup is held to its own source tree's W")
    args = ap.parse_args(argv)
    roots = [ac.read_attempt_root(p) for p in args.roots] if args.roots else ac.discover_attempt_roots(args.stores) if args.stores else None
    if roots is None:
        refuse("one of --stores / --roots is required")
    pins = ac.read_json(args.arditi_pins) if args.arditi_pins else None
    carried = ac.carry_source_ws(ac.read_json(args.carry_over)) if args.carry_over else None
    rows, legs = build(roots, pins=pins, pin_key=args.arditi_pin_key, W_expected=args.W, carried=carried)
    ac.write_csv(args.out, rows, COLUMNS)
    legs_out = args.legs_out or (Path(args.out).parent / "engine_identity_legs.json")
    ac.write_json(legs_out, legs)
    n_fail = sum(1 for v in legs.values() if v["verdict"] != "PASS")
    print(f"[engine-identity] {len(rows)} standup row(s), {len(legs)} leg(s): {len(legs) - n_fail} PASS, {n_fail} FAIL → {args.out}, {legs_out}")
    for leg, v in legs.items():
        if v["verdict"] != "PASS":
            print(f"  FAIL {leg}: " + "; ".join(v["problems"][:4]))
    return ac.EXIT_OK if n_fail == 0 else ac.EXIT_FAILED


if __name__ == "__main__":
    sys.exit(ac.run_main(main))
