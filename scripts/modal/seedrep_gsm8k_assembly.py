#!/usr/bin/env python
"""seedrep_gsm8k_assembly.py -- the GSM8K rung's QUEUE-DERIVED INSPECT ASSEMBLY (EVERY ladder leg incl.
GSM8K escalates at > 1 % at cap up to 16,384). Run by the `run_rung` container UNDER THE tinycap CLIENT VENV (it imports the
tree's `sisteer` and, through `inspect_rung.classify_store` -> `store_texts` -> `inspect_served.reusable_log`, `inspect_ai.log` -- BEFORE any
escalation decision; the driver venv carries no inspect_ai, so `gsm8k_assembly_argv` refuses without a client python), after
its 24,832 x 21 thinking-ON engine is ready and BEFORE any Inspect child:

  1. WAIT for the queue `(gsm8k, <view>, 16384)` to CLOSE: `reconcile_queue` over the DECLARED base producers (their ACCEPTED attempts'
     inventories, every queue file verified) -- `CLOSED.json` lands when every producer is closed; the deadline is the container's;
  2. SNAPSHOT the results Dict (`snapshot_kv`; GSM8K has no elected draws -- the consumer never pre-draws it -- but the assembly reads one);
  3. per (arm, s) SLOT: locate the ACCEPTED base attempt's source store (`<accepted root>/exp/sampled/gsm8k_thinking_on_cap8192__s<s>/<cell>`) and
     run `rung_redraw.gsm8k_subset_from_queue`: `inspect_rung.classify_store` over that store (the ONE at-cap authority: the > 1 % trigger,
     eligible = at-cap minus loops, the transport check) cross-checked against the CLOSED queue's accepted entries of EXACTLY that slot (an entry
     the source never filed at cap REFUSES; a loop the eligible set excludes is `extra`; an eligible item the queue lacks is `missing_from_queue`);
  4. for every slot that ESCALATES write its `rung_subset.json` (`rung_subset/1`, validated by `contracts.validate_item_subset`, bound to the source
     store's identity hash + cap) where the rung child's `--item-subset` names it; a slot below the trigger is `not_triggered` (no child).

The verdict (`--out`) is what the container reads to spawn exactly the escalated slots' children; ANY refusal makes the verdict `ok: false` and the
row fails by name (never a partial rung). Exit 0 = assembled; 3 = the queue did not close by the deadline (the rung is INCOMPLETE, like the
consumer's EXIT_INCOMPLETE); 2 = a refusal. `--memory-kv-dir` swaps the Modal Dicts for JSON files (tests, dry runs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedrep_launch_plan as lp  # noqa: E402

VERDICT_SCHEMA = "seedrep_gsm8k_assembly/1"
EXIT_ASSEMBLED, EXIT_REFUSED, EXIT_NOT_CLOSED = 0, 2, 3
LADDER_LIB_REL = "scripts/modal/seedrep_ladder.py"
RUNG_REDRAW_REL = "scripts/sdfbase/seedrep/rung_redraw.py"


def _load(name: str, path: Path):
    mod = sys.modules.get(name)
    if mod is not None and Path(getattr(mod, "__file__", "")).resolve() == Path(path).resolve():
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_peers(tree: Path):
    """The protocol module (registered as `seedrep_ladder`, the name `rung_redraw._load_ladder` shares) + the consumer / seams module, by path."""
    tree = Path(tree)
    src = str(tree / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    L = _load("seedrep_ladder", tree / LADDER_LIB_REL)
    RR = _load("seedrep_rung_redraw", tree / RUNG_REDRAW_REL)
    return L, RR


class JsonKV:
    """A JSON-file KV with the KV protocol (`put(skip_if_exists)` / get / pop / contains / items) for tests and dry runs -- one process, never
    a distributed store."""

    def __init__(self, path: Path, name: str = "") -> None:
        self.path, self.name = Path(path), name or Path(path).stem
        self._d: Dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._d, indent=1, sort_keys=True))
        tmp.replace(self.path)

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self._d:
            return False
        self._d[key] = value
        self._flush()
        return True

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        v = self._d.pop(key, default)
        self._flush()
        return v

    def contains(self, key: str) -> bool:
        return key in self._d

    def items(self):
        return iter(dict(self._d).items())


def kv_for(L, name: str, memory_kv_dir: Optional[Path]):
    if memory_kv_dir:
        return JsonKV(Path(memory_kv_dir) / f"{name}.json", name)
    return L.ModalDictKV.from_name(name)


def wait_closed(L, vol, accepted_kv, spec, *, results_root: str, deadline_s: float, poll_s: float, clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], Any] = time.sleep, log: Callable[[str], Any] = print) -> Optional[dict]:
    """Reconcile the queue every ``poll_s`` until `CLOSED.json` exists (every declared producer's accepted attempt verified at ITS root -- this
    app's `shard<NN>` spelling, `lp.producer_attempt_root_of`) or the deadline."""
    t0 = clock()
    root_of = lp.producer_attempt_root_of(results_root)
    while True:
        status = L.reconcile_queue(vol, accepted_kv, spec, attempt_root_of=root_of)
        closed = L.read_closed(vol, spec)
        if closed is not None:
            return {"closed": closed, "status": status}
        states = {p: s.get("state") for p, s in (status.get("producers") or {}).items()}
        log(f"[gsm8k-assembly] queue {spec.ladder_root} open: producers {states}")
        if clock() - t0 >= deadline_s:
            return None
        sleep(poll_s)


def local_path(vol_root: Path, vol_path: str) -> Path:
    """A `/vol/...` path (the plan's spelling) under the volume's LOCAL root (in-container `/vol`; a test's tmp dir)."""
    vp = str(vol_path)
    rel = vp[len(lp.VOL_PATH) + 1:] if vp.startswith(lp.VOL_PATH + "/") else vp.lstrip("/")
    return Path(vol_root) / rel


def source_store_for(L, *, vol_root: Path, accepted_kv, spec, results_root: str, slot: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The ACCEPTED base attempt whose root holds the slot's source store (`<root>/<source_out_rel>/cells`), else None."""
    for p in spec.producers:
        aid = L.accepted_attempt(accepted_kv, p.key)
        if not aid:
            continue
        root_vol = lp.attempt_root_for(results_root, leg=spec.leg, view=spec.view, stage=lp.STAGE_BASE, shard=int(p.shard), attempt_id=aid, kind="base")
        src = local_path(vol_root, root_vol) / slot["source_out_rel"]
        if (src / "cells").is_dir():
            return {"attempt_id": aid, "shard": p.shard, "attempt_root": root_vol, "source_out_root": str(src)}
    return None


def assemble(*, L, RR, vol_root: Path, spec, accepted_kv, results_kv, results_root: str, attempt_root: Path, slots: Sequence[Mapping[str, Any]],
             snapshot_path: Path, classify_store: Optional[Callable[..., tuple]] = None, log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """Steps 2-4 over a CLOSED queue -> the verdict document (pure over the volume + KVs; `classify_store` injectable for tests -- the
    default is `inspect_rung.classify_store`, the seam's own)."""
    from sisteer.capeval.tinycap import contracts as C  # the tree's (the CLIENT venv); validates the subset the child reads
    L.snapshot_kv(results_kv, Path(snapshot_path), name=getattr(results_kv, "name", "results"))
    ladder_root = local_path(vol_root, f"{lp.VOL_PATH}/{spec.ladder_root}")
    out: Dict[str, Any] = {"schema": VERDICT_SCHEMA, "queue_root": spec.ladder_root, "cap": int(spec.cap_next), "source_cap": int(spec.base_cap),
                           "trigger_threshold": None, "slots": [], "ok": True, "n_escalate": 0, "n_not_triggered": 0, "n_refused": 0, "assembled_utc": lp._utc()}
    try:
        from sisteer.capeval.tiers import inspect_rung as IR
        out["trigger_threshold"] = float(IR.GSM8K_TRIGGER_THRESHOLD)
    except Exception:  # noqa: BLE001
        IR = None
    for slot in slots:
        rec: Dict[str, Any] = {"arm": slot["arm"], "s": int(slot["s"]), "cell": slot["cell"], "escalate": False, "exit": None, "subset_path": None, "refused": None}
        src = source_store_for(L, vol_root=vol_root, accepted_kv=accepted_kv, spec=spec, results_root=results_root, slot=slot)
        if src is None:
            rec["refused"] = f"no ACCEPTED base attempt of {spec.leg}/{spec.view} holds the source store {slot['source_out_rel']}"
            out["slots"].append(rec)
            continue
        rec.update(source_attempt_id=src["attempt_id"], source_attempt_root=src["attempt_root"], source_out_root=src["source_out_root"])
        try:
            kw = {"classify_store": classify_store} if classify_store is not None else {}
            record, subset, verdict = RR.gsm8k_subset_from_queue(ladder_root, snapshot_path, source_out_root=src["source_out_root"], arm=slot["arm"],
                                                                 seed_base=int(slot["s"]), log=log, **kw)
        except Exception as e:  # noqa: BLE001 -- a LadderRefusal / InspectRungError / a torn store: the slot is REFUSED, the row fails by name
            rec["refused"] = f"{type(e).__name__}: {e}"
            out["slots"].append(rec)
            continue
        rec.update(exit=verdict.get("exit"), n=record.get("n"), n_at_cap=record.get("n_at_cap"), share_at_cap=record.get("share_at_cap"),
                   n_eligible=verdict.get("n_eligible"), n_queued=verdict.get("n_queued"), missing_from_queue=verdict.get("missing_from_queue"),
                   extra_in_queue=verdict.get("extra_in_queue"), item_ids_at_cap=sorted(str(r["item_id"]) for r in record.get("item_facts", []) if r.get("at_cap")))
        if subset is None:
            rec["escalate"] = False
            out["slots"].append(rec)
            continue
        subset_path = Path(attempt_root) / slot["subset_rel"]
        doc = dict(subset)
        C.validate_item_subset(doc, where=f"{slot['arm']} s{slot['s']} rung subset")
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = subset_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        tmp.replace(subset_path)
        rec.update(escalate=True, subset_path=str(subset_path), n_subset=len(doc["ids"]), subset_sha16=C.item_subset_sha16(doc) if hasattr(C, "item_subset_sha16") else None)
        out["slots"].append(rec)
    out["n_escalate"] = sum(1 for s in out["slots"] if s["escalate"])
    out["n_refused"] = sum(1 for s in out["slots"] if s["refused"])
    out["n_not_triggered"] = sum(1 for s in out["slots"] if not s["escalate"] and not s["refused"])
    out["ok"] = out["n_refused"] == 0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--vol-root", type=Path, default=Path(lp.VOL_PATH))
    ap.add_argument("--results-root", required=True, help="the plan's results_root (/vol/seedrep/<W7>/exp/seedrep_battery_modal)")
    ap.add_argument("--W7", required=True)
    ap.add_argument("--wave-uuid", required=True)
    ap.add_argument("--leg", default="gsm8k")
    ap.add_argument("--view", required=True)
    ap.add_argument("--cap-next", type=int, default=16384)
    ap.add_argument("--producers", required=True, help="comma-separated feeder shards (the base shards)")
    ap.add_argument("--accepted-dict", required=True)
    ap.add_argument("--results-dict", required=True)
    ap.add_argument("--attempt-root", type=Path, required=True, help="the rung attempt's LOCAL root (the subsets are written under it)")
    ap.add_argument("--slots", required=True, help="JSON list of {arm, s, cell, source_out_rel, subset_rel}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--deadline-s", type=float, default=6 * 3600)
    ap.add_argument("--memory-kv-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    L, RR = load_peers(a.tree)
    lp.check_wave_uuid(a.wave_uuid)
    shards = [s.strip() for s in a.producers.split(",") if s.strip()]
    spec = L.QueueSpec.build(a.W7, a.wave_uuid, a.leg, a.view, "gsm8k", int(a.cap_next), shards)
    vol = L.LocalVolume(Path(a.vol_root))
    accepted_kv = kv_for(L, a.accepted_dict, a.memory_kv_dir)
    results_kv = kv_for(L, a.results_dict, a.memory_kv_dir)
    closed = wait_closed(L, vol, accepted_kv, spec, results_root=a.results_root, deadline_s=float(a.deadline_s), poll_s=float(a.poll_s))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if closed is None:
        a.out.write_text(json.dumps({"schema": VERDICT_SCHEMA, "queue_root": spec.ladder_root, "closed": False, "ok": False, "slots": [],
                                     "why": f"the queue did not CLOSE within {a.deadline_s} s"}, indent=1) + "\n")
        return EXIT_NOT_CLOSED
    verdict = assemble(L=L, RR=RR, vol_root=Path(a.vol_root), spec=spec, accepted_kv=accepted_kv, results_kv=results_kv, results_root=a.results_root,
                       attempt_root=a.attempt_root, slots=json.loads(a.slots), snapshot_path=(a.snapshot or a.attempt_root / "ladder" / "results_snapshot.json"))
    verdict["closed"] = True
    a.out.write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    print(f"[gsm8k-assembly] {verdict['n_escalate']} slot(s) escalate, {verdict['n_not_triggered']} below the trigger, {verdict['n_refused']} refused -> {a.out}")
    return EXIT_ASSEMBLED if verdict["ok"] else EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
