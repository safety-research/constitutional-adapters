"""The bcb-bfcl-200-widen analysis (``scripts/bcb_bfcl_200_analysis.py``) on
ROSTER-DRIVEN, PRODUCTION-SHAPED wave fixtures: the shared store builders of
``tests/test_tinycap_ladder.py`` (rows from ``ItemResult.extra()``, the store
side's own writers, the lane's real pins) re-shaped to the widen lane's
producer — the manifest's PROTOCOL ``sampling`` block (`by_task` with executed /
protocol / effective; run-level scalars null; `served.sampling_clamp_override`),
the hashed `sampling_by_task` map, the driver's `provenance.artifact_sha256`
with the route digests NULL as the first lane's real manifests carry them, the
launch-time `lane_roster.json` sidecar per store root, the object sidecars under
`_objects/`, one launcher run id per wave in the attempt pointers.

Covers: the roster reader and its refusals; a four-cell gemma instruct-direct
wave (none, prompted, two batch-8 fitted objects) paired per steered cell with
the reading rule; the generation guard (batch-8 vs own `none` PASSES; a
relabelled sidecar — digest stale OR recomputed —, a sha-mismatched sidecar, a
missing sidecar, a sidecar naming a roster sha no revision has all REFUSE: the
label is RE-DERIVED from the roster at the sidecar's roster sha, found in the
roster's git history when the file has moved on; two fitted generations refuse
pairing; `none` needs no sidecar; roster drift is reported); same-wave provenance
with the SUFFIX as the durable identity (a resumed cell's second attempt is
accepted when bound to the suffix and swept; a pointer from another suffix / sha
/ cell, or an attempt without its sweep certificate, refuses); the protocol record
read literally (a clamped BFCL, a missing override — ONE problem —, a card store
under a protocol roster, a protocol/card mix all refuse); the roster gating the
stores (an unlisted cell, a blocked leg filed, a missing cell); fit layers per
cell from the object's sidecar with the roster cross-check; the shared
`project_identity` consumer on full→single-leg fixtures (the ladder's, no local
copy) with the differing values in the refusal; paper rows in the notebook's
schema (48 columns: `object_artifact_sha256` + `thinking` + `held_out` added) with lane_kind /
view per wave, provisional marks, `decode` from the record beginning with the
literal words "benchmark protocol" on EVERY row (a not-run row: the declared
protocol, labelled), queued / blocked rows, the APPEND semantics of the lane
parquet under a lock (two processes), and `--paper-rows-replace` replacing ONCE
for all waves; `--pull` validating the roster first; the cross-wave summary, its
figures and the results doc (full and skeleton) through the ONE final-reading
selector; the CLI end to end with W&B disabled. Pure: no Modal, no W&B, no
network — matplotlib on Agg; git only on a tmp repo."""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
import yaml  # noqa: E402

import test_tinycap_ladder as F  # noqa: E402  - the shared fixture builders (the repo's sibling-test convention)
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import rung_probe as RP  # noqa: E402
from sisteer.capeval.tinycap import widen_roster as W  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bcb_bfcl_200_analysis.py"


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


An = _load(REPO / "scripts" / "bcb_bfcl_200_analysis.py", "bcb_bfcl_200_analysis")
L, M = An.L, An.M
BCB, BFCL = F.BCB, F.BFCL
NONE = "none"
PROMPTED = "prompted__safety_refusal"
CONST_LORA, CONST_PSR = "constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"
DIFF_LORA, DIFF_PSR = "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"
SHA = {CONST_LORA: "4c1a5fc3b382ddb86f9dce20ef7d31dac735b516014017d19fad0c74c0092353",
       CONST_PSR: "fdab3ade9b0d8d627ce2519b0fa72552fd3fa44537d8ea12e6f9727f11808b5b",
       DIFF_LORA: "be93b469a647248c68b48482c42e5af8f24da0993ba789f35e7a3aeeb2663391",
       DIFF_PSR: "3df21c4a3f51a46d124ad6a1bed33c207aaf38931ecb00a4d5ceb0079a124cc9",
       "qwen_diff": "e24b4aaa8f2e431eb4d9c368516f832f60c8e62637782c0b3f9283d7d2b440ff"}
QWEN_HF, QWEN_REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CLAMP = {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"}
T0 = 1_789_000_000.0            # a synthetic epoch clock

# --------------------------------------------------------------------------- #
# the wave world                                                               #
# --------------------------------------------------------------------------- #


ROSTER_PATH = REPO / "configs/capeval/bcb_bfcl_200_widen_roster.yaml"


def _obj(cell: str, *, gen: str = "final", layers: Any = "0-41", kind: str = "lora", sidecar: str = "train_budget_meta.json",
         superseded: Optional[str] = None, staged: Optional[str] = None, sha: Optional[str] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"staged_id": staged or f"itdoc__{cell}", "sha256": sha or SHA.get(cell, "a" * 64), "kind": kind, "sidecar": sidecar,
                         "fit": {"layers": layers}, "object_generation": gen}
    if superseded:
        d["superseded_by_key"] = superseded
    return d


#: the PSR entries the SHIPPED roster blocks (PSR is out of the paper by default)
PSR_ENTRIES: Tuple[Tuple[str, str], ...] = (("gemma4e4b_it", "const_psr"), ("gemma4e4b_main", "diff_psr"))
#: the paper's constitution-SI arm (one of its six arms): a runnable PROMPTED cell of the gemma instruct-direct wave
CONST_SI = "prompted__si_constitution"


#: the gemma4e4b_it entries the FIXTURE parks back to `queued` (their pre-#107 texts): the fixture worlds build stores for the
#: it wave's cell set as it stood before Scale staged the batch-64 pair (2026-09-14) — no fixture store, pin or sidecar exists
#: for the two new cells, so `roster_doc()` re-parks them by default (`eb64="runnable"` reads the roster as shipped)
EB64_PARKED = {"ctl_lora": ("no instruct-direct fit exists; it is part of Scale's batch-64 fit batch (D7)",
                            "Scale's 'ids + shas + pins entry' message for gemma4e4b_it_doc_eb64 -> set runnable (generation final)"),
               "diff_lora": ("no instruct-direct derived object (same batch)", "as ctl_lora (derived pair staged)")}


#: the qwen BFCL legs PARKED back to `queued` for the fixture worlds (they hold no qwen BFCL store): the shipped roster has
#: run them RUNNABLE since 2026-09-23 (the live half of the S5 probe PASSED at 80df464)
QWEN_BFCL_PARKED = {"status": "queued",
                    "reason": "S5 half-verified (fixture park — the shipped roster is runnable since W3-17): the LIVE half's store is not in this world",
                    "unblock": "fixture only: roster_doc(bfcl_qwen='shipped') reads the checked-in state"}


def roster_doc(*, bfcl_qwen: str = "queued", psr: str = "blocked", eb64: str = "queued") -> Dict[str, Any]:
    """The CHECKED-IN roster (`configs/capeval/bcb_bfcl_200_widen_roster.yaml`), deep-copied so a test may
    mutate it; by default every qwen AND nemotron wave's BFCL leg is PARKED back to `queued` (:data:`QWEN_BFCL_PARKED`; the
    fixture worlds hold no qwen / nemotron BFCL store — the shipped roster has run qwen's runnable since 2026-09-23 and
    nemotron's since its own PASS at 8365992 the same day), ``bfcl_qwen="shipped"`` leaves
    the legs as checked in and ``bfcl_qwen="runnable"`` forces the main wave's leg runnable; ``psr="runnable"`` REINSTATES the two PSR cells the
    shipped roster blocks — store-building variants the shipped state never has (the object blocks and the
    batch-8 `rerun_trigger` ride the blocked entries intact, so the flip is exactly the earlier roster). By default the
    gemma4e4b_it batch-64 control LoRA / LoRA subtraction (runnable on the shipped roster since 2026-09-14) are PARKED back to
    `queued` with their pre-staging texts (:data:`EB64_PARKED`): the fixture worlds have no stores for them; ``eb64="runnable"``
    leaves them as shipped."""
    import copy
    doc = copy.deepcopy(yaml.safe_load(ROSTER_PATH.read_text(encoding="utf-8")))
    if eb64 == "queued":
        for eid, (reason, unblock) in EB64_PARKED.items():
            e = doc["waves"]["gemma4e4b_it"]["entries"][eid]
            e.pop("object", None)
            e.pop("note", None)
            e.update(status="queued", reason=reason, unblock=unblock)
    if bfcl_qwen == "queued":
        # ... and the nemotron waves' since nemotron_h's own live PASS at 8365992 (2026-09-23): the fixture worlds hold no
        # nemotron BFCL store either, so the same park applies to every probe family's waves
        for w in doc["waves"].values():
            if w.get("family") in ("qwen35_4b", "nemotron3nano4b"):
                w["legs"]["bfcl_200"] = dict(QWEN_BFCL_PARKED)
    elif bfcl_qwen == "runnable":
        doc["waves"]["qwen35_4b_main"]["legs"]["bfcl_200"] = "runnable"
    if psr == "runnable":
        for wave, eid in PSR_ENTRIES:
            e = doc["waves"][wave]["entries"][eid]
            e["status"] = "runnable"
            e.pop("reason", None)
            e.pop("unblock", None)
    return doc


def fixture_roster(**kw: str) -> Dict[str, Any]:
    """The roster the analysis FIXTURE worlds are built on: the shipped roster with the PSR cells reinstated, so every
    gemma wave keeps TWO fitted cells (the ladder's per-cell certificates and wave record, the batch-8 pair, the
    two-object pairing guards all need them). Tests that pin the SHIPPED state read `roster_doc()` instead."""
    return roster_doc(psr="runnable", **kw)


def write_roster(path: Path, doc: Optional[Mapping[str, Any]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(doc or roster_doc()), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


#: a FIXED clock for the fixture's git commits, so a commit of the same roster bytes has the same sha in every
#: test process (the store's producer tree = that commit: the analysis reads the roster AT it)
GIT_DATE = "2026-09-11T20:00:00+0000"


def _git(*args: str, cwd: Path) -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": GIT_DATE, "GIT_COMMITTER_DATE": GIT_DATE, "GIT_CONFIG_NOSYSTEM": "1"}
    r = subprocess.run(["git", "-C", str(cwd), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, env=env)
    assert r.returncode == 0, (args, r.stdout, r.stderr)
    return r.stdout.strip()


def commit_roster(path: Path, *, message: str = "roster") -> str:
    """The lane keeps the roster yaml in git: commit ``path`` in a fresh repo at its parent (or add a revision
    to one already there) and return the COMMIT sha — the fixture stores' `producer.tree_sha`, the commit the
    analysis reads the roster at (`git show <tree_sha>:roster.yaml`). Deterministic: fixed identity and clock."""
    if not (path.parent / ".git").exists():
        _git("init", "-q", cwd=path.parent)
    _git("add", "--", path.name, cwd=path.parent)
    _git("commit", "-q", "--no-verify", "--allow-empty", "-m", message, cwd=path.parent)
    return _git("rev-parse", "HEAD", cwd=path.parent)


def _deterministic_roster_commit() -> str:
    """The sha of the fixture's producer commit (the checked-in roster committed under `lane/roster.yaml` at the
    fixed clock) — computed once per process in a scratch repo, so the module's suffixes and run ids can carry it."""
    scratch = Path(tempfile.mkdtemp(prefix="bcbw-roster-"))
    try:
        return commit_roster(write_roster(scratch / "lane" / "roster.yaml", fixture_roster()))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


#: the launched tree = the roster's producer commit: the suffix, every run id, every store's producer tree and every
#: attempt pointer carry its sha7 (the durable wave identity); every fixture store is produced "at" this commit
TREE = _deterministic_roster_commit()
SHA7 = TREE[:7]
SUFFIX_IT, SUFFIX_MAIN, SUFFIX_QWEN = f"bcbw-gemma4e4b_it-{SHA7}", f"bcbw-gemma4e4b_main-{SHA7}", f"bcbw-qwen35_4b_main-{SHA7}"
RUN_IT, RUN_MAIN, RUN_QWEN = f"{SHA7}-11111111", f"{SHA7}-22222222", f"{SHA7}-33333333"


def protocol_block(tasks: Sequence[str], *, bfcl_executed: float = 0.001, clamp: bool = True) -> Dict[str, Any]:
    """The manifest `sampling` block the driver writes under `protocol`."""
    by_task = {}
    for t in tasks:
        temp = 0.0 if t == BCB else bfcl_executed
        by_task[t] = {"executed": {"temperature": temp, "top_p": 1.0, "top_k": 0}, "protocol_temperature": (0.0 if t == BCB else 0.001),
                      "executed_reason": ("equal" if t == BCB else ("equal (clamp override applied)" if clamp else "CLAMPED to 0.01 by vLLM 0.25.1")),
                      "effective": {"temperature": temp, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0},
                      "source": ("BigCodeBench's reference protocol: greedy pass@1" if t == BCB else "BFCL's generation script default 0.001; D13 override")}
    return {"mode": "protocol", "model_key": "gemma4e4b", "temperature": None, "top_p": None, "top_k": None, "samples_by_task": {t: 1 for t in tasks},
            "seed_rule": C.SEED_RULE_INSPECT, "source": "; ".join(f"{t}: {by_task[t]['source']}" for t in tasks), "by_task": by_task}


def protocolize_store(store_dir: Path, *, tasks: Sequence[str], served_sha: Optional[str], cell: str, kind: Optional[str], arm: Optional[str],
                      bfcl_executed: float = 0.001, clamp: bool = True, null_route_digest: bool = True, model: Optional[Tuple[str, str]] = None) -> None:
    """Re-shape a fixture store's manifest to the widen lane's REAL producer: the protocol `sampling` block, the hashed
    `sampling_by_task` map with the run-level scalars null, `served.sampling_clamp_override`, the driver's
    `provenance` block with `artifact_sha256`, and — as the first lane's manifests carry them — NULL route digests."""
    p = store_dir / "manifest.json"
    m = json.loads(p.read_text())
    m[C.MANIFEST_KEY_SAMPLING] = protocol_block(tasks, bfcl_executed=bfcl_executed, clamp=clamp)
    tc = m[C.MANIFEST_KEY_TINYCAP]
    for k in (C.IDENTITY_KEY_TEMPERATURE, C.IDENTITY_KEY_TOP_P, C.IDENTITY_KEY_TOP_K):
        tc[k] = None
    tc["sampling_by_task"] = {t: dict(m[C.MANIFEST_KEY_SAMPLING]["by_task"][t]["executed"]) for t in tasks}
    if clamp:
        tc["served"]["sampling_clamp_override"] = dict(CLAMP)
    m["kind"], m["arm_name"] = kind, arm
    m[C.MANIFEST_KEY_PROVENANCE] = {"arm": arm, "artifact_sha256": served_sha, "cell_id": cell, "class": ("A" if served_sha else "baseline"),
                                    "code_version": "cell_controls_v1", "intervention_hash": m["intervention_hash"], "model": m["model_id"],
                                    "model_revision": m["model_revision"]}
    if null_route_digest:
        m[C.MANIFEST_KEY_ROUTE][C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] = None
        m[C.MANIFEST_KEY_ROUTE][C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL] = None
    if model:
        m["model_id"], m["model_revision"] = model
        m[C.MANIFEST_KEY_PROVENANCE]["model"], m[C.MANIFEST_KEY_PROVENANCE]["model_revision"] = model
    p.write_text(json.dumps(m, sort_keys=True))


LAUNCHED_AT = "2026-09-11T20:00:00+00:00"


def write_sidecar(cell_root: Path, *, roster: Mapping[str, Any], roster_path: Path, wave: str, entry_id: str,
                  mutate: Optional[Callable[[Dict[str, Any]], None]] = None, launched_at: str = LAUNCHED_AT) -> Path:
    """The launch-time `lane_roster.json` through the PRODUCER's writer (`widen_roster.snapshot`: the roster
    FILE's sha, the entry's generation / superseding key / object sha, the launch time, the digest); ``mutate``
    edits the block the way an adversary who re-digests properly would (the digest is recomputed after it)."""
    block = W.snapshot(roster, wave, entry_id, roster_sha256=W.roster_sha256(roster_path), launched_at=launched_at)
    if mutate is not None:
        mutate(block)
        block["snapshot_sha256"] = W.snapshot_digest(block)
    p = cell_root / An.LANE_ROSTER_SIDECAR
    p.write_text(json.dumps(block, sort_keys=True))
    return p


def write_object_sidecar(rung1: Path, cell: str, *, filename: str, layers: Sequence[int], object_sha: Optional[str] = None, derived: bool = False) -> Path:
    d = rung1 / An.OBJECTS_SUBDIR / cell
    d.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = ({"layers": list(layers), "object_sha256": object_sha, "kind": "lora_subtraction", "model_id": "Qwen/Qwen3.5-4B-Base", "positions": "all"}
                           if derived else {"target_layers": list(layers), "cell_id": cell, "model_id": F.HF, "positions_at_fit": "all", "arm": "lora_r4_ce"})
    p = d / filename
    p.write_text(json.dumps(doc))
    return p


def container_out_dir(cell: str, *, model_slug: str, suffix: str, mode: str = "off") -> str:
    """The store root a container writes into (`cfg["out_dir"]` + the out suffix): the pointer's `out_dir`."""
    return f"{C.cell_out_dir(C.VOL_GEN_ROOT, cell, mode, model_slug=model_slug)}/{suffix}"


def sandbox_and_pointer(cell_root: Path, *, run_id: str, cell: str, suffix: str, model_slug: str, fcid: Optional[str] = None, n: int = 6,
                        controls: Sequence[str] = ("bigcodebench_200", "bfcl_200"), tree_sha: str = TREE, sweep: bool = True,
                        started_at: float = T0) -> str:
    """One launcher ATTEMPT of a cell in the producer's shape: `watch/sandboxes_<fcid>.jsonl` (EMPTY — zero bytes — when
    ``n`` is 0: the launcher creates the record before publishing the pointer, so a standup that fails before the driver
    leaves an empty one), the attempt pointer (`tinycap_served.POINTER_FIELDS`: the owner tuple, the CONTAINER's out_dir =
    this suffix's store root, the tree) and the container's sweep certificate (`sandbox_scope.sweep_owned`'s record).
    Returns the fcid."""
    fcid = fcid or f"fc-{cell[:6]}"
    owner = {"run_id": run_id, "cell": cell, "function_call_id": fcid}
    lines = []
    for k in range(n):
        sid = f"sb-{cell[:4]}-{fcid[-3:]}-{k:03d}"
        lines.append({"ts": started_at + 60 * k, "event": "created", "sandbox_id": sid, "owner": owner, "task": "bigcodebench", "sample_id": str(k)})
        if k < n - 1:
            lines.append({"ts": started_at + 60 * k + 30 + 5 * k, "event": "terminated", "sandbox_id": sid, "owner": owner, "verified_by": "poll" if k % 2 else "relist"})
    watch = cell_root / "watch"
    watch.mkdir(exist_ok=True)
    out_dir = container_out_dir(cell, model_slug=model_slug, suffix=suffix)
    (watch / f"sandboxes_{fcid}.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    (watch / f"attempt_{fcid}.json").write_text(json.dumps({
        "run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"{cell}-off", "model": model_slug, "mode": "off",
        "out_dir": out_dir, "controls": list(controls), "tree_sha": tree_sha, "sandbox_record": f"{out_dir}/watch/sandboxes_{fcid}.jsonl",
        "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": started_at}))
    if sweep:
        (watch / f"sweep_{fcid}.json").write_text(json.dumps({"swept_at": started_at + 3600, "by": "container_finally", "owner": owner, "ids_terminated": [],
                                                              "left": [], "could_not_terminate": [], "still_live": [], "tags_unreadable": [],
                                                              "verified_empty": True, "rounds": 1, "n_listed_first_round": 0}))
    return fcid


def watcher_sweep(state_root: Path, *, run_id: str, cell: str, fcid: str, listed: int = 0, terminated: Sequence[str] = (),
                  verified_empty: bool = True, swept_at: float = T0 + 7200) -> Path:
    """The EXTERNAL watcher's sweep certificate for an attempt (`tinycap_sandbox_watch`: `<state root>/<run_id>/sweep_<fcid>.json`,
    `sandbox_scope.sweep_owned`'s record) — what covers an attempt whose container died before its own `finally`."""
    p = state_root / run_id / f"sweep_{fcid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"swept_at": swept_at, "by": "watcher", "owner": {"run_id": run_id, "cell": cell, "function_call_id": fcid},
                             "ids_terminated": list(terminated), "left": [], "could_not_terminate": [], "still_live": [], "tags_unreadable": [],
                             "verified_empty": verified_empty, "rounds": 1, "n_listed_first_round": int(listed)}, sort_keys=True))
    return p


def wave_values(cell: str):
    """A deterministic paired pattern per cell (distinct per steered cell, so each contrast differs)."""
    salt = sum(map(ord, cell)) % 5

    def values(c: str, task: str, i: int) -> float:
        base = i % 3 != 0
        if c == NONE:
            return float(base)
        if base and (i + salt) % 7 == 1:
            return 0.0
        if not base and (i + salt) % 11 == 5:
            return 1.0
        return float(base)
    return values


def make_wave(root: Path, roster: Mapping[str, Any], wave: str, *, roster_path: Path, run_id: Optional[str] = None, capped: Optional[Mapping] = None,
              values=None, sidecars: bool = True, object_sidecars: bool = True, null_route_digest: bool = True, bfcl_executed: float = 0.001,
              clamp: bool = True, protocol: bool = True, served: Tuple[int, int] = F.SERVED_RUNG1, combined: bool = True,
              tree_sha: str = TREE) -> Path:
    """A pulled rung-1 dir for ``wave`` at ``root`` (whose NAME is the store suffix, `bcbw-<wave>-<sha7>[-tag]`): every
    runnable cell's store (both runnable legs in ONE store per cell, the normal launch) produced at ``tree_sha`` (the
    roster's committed producer commit), the protocol manifest shape, the launch-time sidecar for every fitted cell
    (the producer's writer over the roster FILE at ``roster_path``), the object sidecars, one launcher attempt per cell
    under one run id (minted at the suffix's sha) across the wave."""
    spec = An.wave_spec(roster, wave)
    run_id = run_id or f"{tree_sha[:7]}-deadbeef"
    capped = dict(capped or {})
    model = (None if spec.served["hf"] == F.HF else (spec.served["hf"], spec.served["rev"]))
    for e in spec.runnable_entries:
        cell = e.cell_id
        vals = values or wave_values(cell)
        by_task = {t: F.rung1_results(cell, t, capped=capped.get((cell, t), ()), values=vals) for t in spec.tasks}
        sha = e.object["sha256"] if e.fitted else None
        kind = {"base": None, "prompted": None}.get(e.kind, (e.object or {}).get("kind"))
        groups = [dict(by_task)] if combined else [{t: r} for t, r in by_task.items()]
        for group in groups:
            store_dir = F.make_multi_store(root, cell, group, caps={t: F.CAP_OFF[t] for t in group}, served=served, limits={}, object_sha256=sha,
                                           tree_sha=tree_sha)
            if protocol:
                protocolize_store(store_dir, tasks=list(group), served_sha=sha, cell=cell, kind=kind, arm=(None if cell == NONE else cell.replace("__", "--")),
                                  bfcl_executed=bfcl_executed, clamp=clamp, null_route_digest=null_route_digest, model=model)
            elif model:
                m = json.loads((store_dir / "manifest.json").read_text()); m["model_id"], m["model_revision"] = model
                (store_dir / "manifest.json").write_text(json.dumps(m, sort_keys=True))
        if sidecars and e.fitted:
            write_sidecar(root / cell, roster=roster, roster_path=roster_path, wave=wave, entry_id=e.entry_id)
        if object_sidecars and e.fitted:
            layers = An._declared_layers(e.object["fit"])
            write_object_sidecar(root, cell, filename=An.sidecar_filename(e), layers=layers, object_sha=sha, derived=e.object["sidecar"] == "derived_meta.json")
        sandbox_and_pointer(root / cell, run_id=run_id, cell=cell, suffix=root.name, model_slug=str(spec.model_row),
                            controls=[An.ATOM_OF_TASK[t] for t in spec.tasks], tree_sha=tree_sha)
    return root


def ledger_with(path: Path, *entries: Mapping[str, Any]) -> Path:
    for e in entries:
        M.append_entry(path, e)
    return path


@pytest.fixture()
def world(tmp_path):
    repo = F.make_pins(tmp_path / "repo")
    roster = write_roster(tmp_path / "lane" / "roster.yaml", fixture_roster())
    assert commit_roster(roster) == TREE        # the producer commit every fixture store names: the analysis reads the roster AT it
    doc = An.load_roster(roster)
    it = make_wave(tmp_path / SUFFIX_IT, doc, "gemma4e4b_it", roster_path=roster, run_id=RUN_IT,
                   capped={(CONST_LORA, BCB): set(F.CAPPED_BCB), (NONE, BCB): {F.CAPPED_BCB[0]}})
    main = make_wave(tmp_path / SUFFIX_MAIN, doc, "gemma4e4b_main", roster_path=roster, run_id=RUN_MAIN)
    ledger = ledger_with(tmp_path / "gpu_ledger.jsonl",
                         {"kind": "decision", "ceiling_h": 25.0, "note": "spec envelope"},
                         {"kind": "forecast", "wave": "gemma4e4b_it", "suffix": SUFFIX_IT, "unit": "smoke", "stage_kind": "smoke", "hours": 0.55},
                         {"kind": "actual", "wave": "gemma4e4b_it", "suffix": SUFFIX_IT, "unit": "smoke", "stage_kind": "smoke", "hours": 0.5},
                         *({"kind": "forecast", "wave": "gemma4e4b_it", "suffix": SUFFIX_IT, "unit": c, "stage_kind": "wave", "hours": h}
                           for c, h in ((NONE, 0.18), (PROMPTED, 0.2), (CONST_LORA, 0.4), (CONST_PSR, 0.4))),
                         {"kind": "forecast", "wave": "gemma4e4b_main", "suffix": SUFFIX_MAIN, "unit": NONE, "stage_kind": "wave", "hours": 0.18})
    return {"repo": repo, "roster": roster, "doc": doc, "it": it, "main": main, "ledger": ledger, "tmp": tmp_path}


def _analyze(world, wave="gemma4e4b_it", rung1=None, **kw):
    kw.setdefault("sandbox_state_root", world["tmp"] / "sandbox-state")        # never the grading host's real watcher state
    return An.analyze(rung1 or world["it"], roster_path=world["roster"], wave_name=wave, repo_root=world["repo"], ledger=world["ledger"], **kw)


def expected_paper_rows(analysis: Mapping[str, Any]) -> int:
    """The paper-row count the roster implies: runnable cells × runnable legs, every non-runnable entry × both legs,
    every runnable cell × every blocked leg (reported by name, never dropped)."""
    entries = analysis["wave"]["entries"]
    n_not_run = sum(1 for e in entries if e["status"] != "runnable")
    return len(analysis["cells"]) * len(analysis["tasks"]) + n_not_run * len(An.TASKS) + len(analysis["cells"]) * len(analysis.get("blocked_legs") or {})


def _edit_manifest(rung1: Path, cell: str, task: str, fn) -> None:
    d = L.store_dir_for(rung1, cell, task)
    m = json.loads((d / "manifest.json").read_text())
    fn(m)
    (d / "manifest.json").write_text(json.dumps(m, sort_keys=True))


# --------------------------------------------------------------------------- #
# the roster                                                                   #
# --------------------------------------------------------------------------- #


def test_wave_spec_reads_the_plans_shape_and_refuses_bad_entries(tmp_path):
    doc = roster_doc(bfcl_qwen="shipped")                     # the SHIPPED roster: PSR is out by default; the constitution SI is in
    it = An.wave_spec(doc, "gemma4e4b_it")
    assert it.cells == W.runnable_cells(doc, "gemma4e4b_it") == [NONE, PROMPTED, CONST_SI, CONST_LORA] and it.steered_cells == [PROMPTED, CONST_SI, CONST_LORA]
    assert it.fitted_cells == [CONST_LORA] and it.tasks == [BCB, BFCL] and it.blocked_legs == {}
    assert {e.entry_id for e in it.queued_or_blocked} >= {"ctl_lora", "diff_psr", "const_psr", "alphasteer_as_fit"} and all(not e.runnable for e in it.queued_or_blocked)
    si = it.entry_of_cell(CONST_SI)
    assert (si.entry_id, si.kind, si.paper_arm, si.fitted) == ("const_si", "prompted", "si_constitution", False) and "six-arm ruling" in si.note
    assert An.paper_arm({"cell_id": CONST_SI, "paper_arm": si.paper_arm}) == ("si_constitution", "prompted", "prompt")
    # the rule: every PSR cell is BLOCKED with the ruling as its reason, its object block (the pin) kept so it stays identifiable
    for wave, eid in PSR_ENTRIES:
        e = next(x for x in An.wave_spec(doc, wave).entries if x.entry_id == eid)
        assert e.status == "blocked" and e.fitted and e.reason.startswith(An.PSR_OUT_RULING_PREFIX) and "PSR is out of the paper by default" in e.reason
        assert e.unblock == "a ruling reinstates PSR" and e.object["sha256"] == SHA[e.cell_id] and W.validate(doc, W.load_pins(), C.MODELS) == []
        assert An.roster_status_label(e.status, e.reason) == An.PSR_OUT_LABEL
    assert An.roster_status_label("blocked", "no verified parser") == "blocked" and An.roster_status_label("queued", None) == "queued"
    main = An.wave_spec(doc, "gemma4e4b_main")
    assert main.lane_kind == "base→instruct" and main.cells == W.runnable_cells(doc, "gemma4e4b_main") == [NONE, DIFF_LORA]   # the yaml's `->` normalised to the paper's arrow
    assert it.entry_of_cell(CONST_LORA).object_generation == "batch8-provisional" and it.entry_of_cell(PROMPTED).kind == "prompted"
    # the fixture variant reinstates PSR (two fitted cells per gemma wave) and still passes the reader's rules
    fx = fixture_roster()
    assert An.wave_spec(fx, "gemma4e4b_it").fitted_cells == [CONST_LORA, CONST_PSR] and An.wave_spec(fx, "gemma4e4b_main").cells == [NONE, DIFF_LORA, DIFF_PSR]
    assert W.validate(fx, W.load_pins(), C.MODELS) == []
    qw = An.wave_spec(doc, "qwen35_4b_main")
    # since 2026-09-23 the qwen BFCL leg is RUNNABLE on the shipped roster — the LIVE half of the
    # S5 probe PASSED at the staged tree 80df464 and its verdict is checked in; the fixture worlds PARK it back to queued
    assert qw.tasks == [BCB, BFCL] and qw.blocked_legs == {}
    parked = An.wave_spec(roster_doc(), "qwen35_4b_main")
    assert parked.tasks == [BCB] and list(parked.blocked_legs) == [BFCL] and parked.blocked_legs[BFCL]["status"] == "queued"
    assert parked.blocked_legs[BFCL]["reason"].startswith("S5 half-verified") and "LIVE" in parked.blocked_legs[BFCL]["reason"]
    # the queued `llama8b` wave is now the two runnable llama8b waves (none + the LoRA subtraction;
    # the it wave instruct-direct, the main wave base->instruct like qwen's). Their BFCL leg is BLOCKED
    # (SKIPPED for Llama — the unblock names the rule, not a re-run path; the pinned template
    # rendered no tools, an instrument failure), so the SHIPPED roster runs them BCB-only
    for wname, slug, kind in (("llama8b_it", "llama8b-f22it", "instruct-direct"), ("llama8b_main", "llama8b-f22", "base→instruct")):
        ll = An.wave_spec(doc, wname)
        assert ll.cells == [NONE, "constv2-mix-full__lora_r4_ce_minus_ctl"] and ll.model_row == slug
        assert ll.tasks == [BCB] and list(ll.blocked_legs) == [BFCL] and ll.lane_kind == kind
        assert ll.blocked_legs[BFCL]["reason"].startswith("instrument failure:") and ll.blocked_legs[BFCL]["unblock"].startswith("NONE — SKIPPED FOR LLAMA by the ruling of 2026-09-13")
    with pytest.raises(An.AnalysisError, match="not in the roster"):
        An.wave_spec(doc, "llama8b")
    with pytest.raises(An.AnalysisError, match="not in the roster"):
        An.wave_spec(doc, "glm46vflash_main")
    bad = roster_doc(); bad["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["cell_id"] = None
    with pytest.raises(An.AnalysisError, match="runnable entry needs a cell_id"):
        An.wave_spec(bad, "gemma4e4b_it")
    bad = roster_doc(); del bad["waves"]["gemma4e4b_it"]["entries"]["ctl_lora"]["unblock"]
    with pytest.raises(An.AnalysisError, match="`reason` AND `unblock`"):
        An.wave_spec(bad, "gemma4e4b_it")
    bad = roster_doc(); del bad["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["object"]["superseded_by_key"]
    with pytest.raises(An.AnalysisError, match="names its superseding key"):
        An.wave_spec(bad, "gemma4e4b_it")
    bad = roster_doc(); bad["waves"]["gemma4e4b_it"]["entries"]["none"]["status"] = "queued"; bad["waves"]["gemma4e4b_it"]["entries"]["none"].update(reason="x", unblock="y")
    with pytest.raises(An.AnalysisError, match="without a runnable `none`"):
        An.wave_spec(bad, "gemma4e4b_it")
    bad = roster_doc(); bad["waves"]["gemma4e4b_it"]["family"] = "glm46vflash"
    with pytest.raises(An.AnalysisError, match="excluded"):
        An.wave_spec(bad, "gemma4e4b_it")


# --------------------------------------------------------------------------- #
# the four-cell gemma instruct-direct wave                                     #
# --------------------------------------------------------------------------- #


def test_gemma_it_wave_pairs_every_steered_cell_against_none(world):
    a = _analyze(world)
    assert a["schema"] == 2 and a["lane"] == "bcb-bfcl-200-widen" and a["wave"]["name"] == "gemma4e4b_it"
    assert (a["wave"]["family"], a["wave"]["lane_kind"], a["wave"]["view"]) == ("gemma4e4b", "instruct-direct", "gemma4e4b_it")
    # the fixture world keeps two fitted cells (PSR reinstated, `fixture_roster`); the cell set is the fixture roster's runnable set
    assert a["cells"] == W.runnable_cells(world["doc"], "gemma4e4b_it") == [NONE, PROMPTED, CONST_SI, CONST_LORA, CONST_PSR]
    assert a["steered"] == [PROMPTED, CONST_SI, CONST_LORA, CONST_PSR] and a["fitted"] == [CONST_LORA, CONST_PSR]
    for task in (BCB, BFCL):
        leg = a["legs"][task]
        assert set(leg["paired"]) == set(a["steered"]) and set(leg["cells"]) == set(a["cells"])
        ka = leg["cells"][NONE]["k"]
        for cell, p in leg["paired"].items():
            kb = leg["cells"][cell]["k"]
            assert abs(p["delta"] - (kb - ka) / 200) < 1e-12 and p["B"] == 10_000 and p["seed"] == 0
            assert p["b_pass_to_fail"] - p["c_fail_to_pass"] == ka - kb and p["reading"].startswith(("no cost detected", "cost detected"))
            assert p["provisional"] == (cell in (CONST_LORA, CONST_PSR))
    assert a["legs"][BCB]["fired"] is True and a["legs"][BFCL]["fired"] is False
    # the generation record: the two fitted cells are batch-8 provisional from the SIDECAR; none / prompted are generation-neutral
    g = a["generations"]
    assert g[CONST_LORA]["object_generation"] == "batch8-provisional" and g[CONST_LORA]["provisional"] and g[CONST_LORA]["superseded_by_key"] == "gemma4e4b_it_doc_eb64"
    assert g[NONE]["object_generation"] is None and g[PROMPTED]["object_generation"] is None and not g[NONE]["fitted"]
    # the served object sha is read from the driver's provenance (the real producer leaves the route digests null)
    assert g[CONST_LORA]["object_sha256"] == SHA[CONST_LORA] and g[CONST_LORA]["object_sha256_source"] == "provenance.artifact_sha256"
    assert a["identity"]["object_sha256"][NONE] is None and a["identity"]["object_sha256"][CONST_PSR] == SHA[CONST_PSR]
    # fit layers per FITTED cell from the object's sidecar, none for the object-free cells
    assert a["fit_layers"][CONST_LORA]["fit_layers"] == "0-41" and a["fit_layers"][CONST_LORA]["fit_layer_count"] == 42 and a["fit_layers"][NONE] is None
    assert a["fit_layers"][CONST_PSR]["layer_key"] == "target_layers"
    # the protocol record, literally
    pr = a["protocol"]
    assert pr["mode"] == "protocol" and pr["clamp_override"] == CLAMP
    assert pr["by_task"][BCB]["executed"]["temperature"] == 0.0 and pr["by_task"][BFCL]["executed"]["temperature"] == 0.001 == pr["by_task"][BFCL]["protocol_temperature"]
    assert all(all(v for v in checks.values()) for checks in pr["check_4_literal"].values())
    assert "T=0.001" in pr["decode_by_task"][BFCL] and "clamp override 0.01->0.001" in pr["decode_by_task"][BFCL] and "T=0.0" in pr["decode_by_task"][BCB]
    # identity: one experiment, projected; same-wave provenance = one run id
    assert a["identity"]["common"]["one_experiment"] is True and a["identity"]["common"]["sampling_mode"] == "protocol"
    assert a["identity"]["common"]["model_row"] == {"slug": "gemma4e4b-f22it", "hf": F.HF, "rev": F.REV, "registered": "gemma4e4b-f22it" in C.MODELS}
    prov = a["identity"]["provenance"]
    assert prov["run_id"] == RUN_IT and prov["run_ids"] == [RUN_IT] and prov["one_wave"] is True and prov["suffix"] == SUFFIX_IT
    assert prov["sha7"] == SHA7 and prov["n_attempts"] == len(a["cells"]) and all(len(v) == 1 and v[0]["sweep_certificate"] for v in prov["attempts_by_cell"].values())
    # the generation labels were RE-DERIVED from the roster at the stores' PRODUCER COMMIT (git show <tree>:roster.yaml)
    assert g[CONST_LORA]["roster_at_launch_source"] == f"git {TREE[:12]}:roster.yaml" and g[CONST_LORA]["roster_sha256_at_launch"] == W.roster_sha256(world["roster"])
    assert a["identity"]["tree_sha"] == TREE
    # the one final-reading rule: BFCL did not fire → rung 1 is final; BCB fired and is not merged → no final reading
    assert a["final"][BFCL] == {**a["final"][BFCL], "source": "rung1", "final": True} and a["final"][BCB]["final"] is False and a["final"][BCB]["cells"] is None
    assert a["final"][BFCL]["paired"] == a["legs"][BFCL]["paired"] and "FIRED" in a["final"][BCB]["why"]
    for task, chk in a["identity"]["checks"].items():
        assert chk["same_instrument"] is True and chk["off_instrument"] == {} and chk["mode"] == "protocol"
        assert "tinycap (projected)" in chk["required_equal"] and any(k.endswith(":intervention_hash") for k in chk["differs"])
    assert a["identity"]["common"]["serving_path_by_cell"] == {c: ("passthrough" if c == NONE else "lens") for c in a["cells"]}
    # breakdowns carry one Δ per steered cell
    bd = a["breakdowns"]["bfcl_by_category"]
    assert bd["n_categories"] == 18 and all(set(r["deltas"]) == set(a["steered"]) for r in bd["rows"])
    assert set(a["quoted"]["bfcl_discordant"]["cells"]) == set(a["steered"]) and set(a["quoted"]["bcb_failures"]["cells"]) == set(a["cells"])
    # the breakdowns / quotes name the rows they were computed from: BFCL did not fire (final: rung 1); BCB fired and is pending (NOT final)
    assert a["breakdowns"]["bfcl_by_category"]["label"] == "final: rung 1" and a["quoted"]["bfcl_discordant"]["final"] is True
    assert a["breakdowns"]["bcb_by_family"]["label"] == "rung 1 — NOT final (ladder pending)" and a["quoted"]["bcb_failures"]["final"] is False
    # cost: this wave's reservations vs the lane's Σ
    gpu = a["cost"]["gpu"]
    assert gpu["present"] and gpu["ceiling_h"] == 25.0 and gpu["ceiling_source"] == "decision"
    assert abs(gpu["wave_h"] - (0.5 + 0.18 + 0.2 + 0.4 + 0.4)) < 1e-9 and abs(gpu["committed_h"] - (gpu["wave_h"] + 0.18)) < 1e-9
    assert gpu["by_wave"] == {"gemma4e4b_it": pytest.approx(1.68), "gemma4e4b_main": pytest.approx(0.18)}
    assert a["cost"]["sandboxes"][NONE]["lifetime_s"]["n"] == 5 and "PARTIAL" in a["cost"]["basis"]


def test_the_projection_is_the_ladders_function_called_directly(monkeypatch):
    """The analysis has no local re-implementation and no `getattr` fallback: `project` calls
    `tinycap_ladder.project_identity` itself (a rename there fails here by name, loudly)."""
    calls = []
    real = L.project_identity

    def spy(block, retained, *, kind="identity", mode=None):
        calls.append((kind, mode, list(retained)))
        return real(block, retained, kind=kind, mode=mode)
    monkeypatch.setattr(L, "project_identity", spy)
    An.project({"tasks": ["bfcl_200"], "thinking_by_task": {BFCL: "off"}, "samples_by_task": {BFCL: 1}}, [BFCL], mode="card", where="x")
    assert calls == [("identity", "card", [BFCL])]
    monkeypatch.delattr(L, "project_identity")
    with pytest.raises(AttributeError):
        An.project({"tasks": []}, [BFCL], mode="card", where="x")


def test_the_analysis_aliases_slice_bs_contracts_instead_of_restating_them():
    """One reader per fact (the simplify pass's altitude review): the sidecar's name / key set / vocabulary, the
    ladder's dir + record names, the object sidecar names, the protocol temperatures, `_dig`."""
    assert An.LANE_ROSTER_SIDECAR is W.SNAPSHOT_FILENAME and An.LANE_ROSTER_FIELDS is W.SNAPSHOT_FIELDS
    assert An.OBJECT_GENERATIONS is W.GENERATIONS and An.ROSTER_STATUSES is W.STATUSES
    assert (An.GENERATION_FINAL, An.GENERATION_BATCH8) == (W.GENERATION_FINAL, W.GENERATION_BATCH8)
    assert An.LADDER_SUBDIR is RP.LADDER_SUBDIR and An.RUNG_ADMISSION_FILENAME is RP.ADMISSION_FILENAME
    from sisteer.eval import train_budget as TBm
    assert An.DEFAULT_SIDECAR == TBm.BUDGET_META_NAME and set(An.DEFAULT_SIDECAR_OF_KIND.values()) == {TBm.DERIVED_META_NAME}
    assert An._dig is L._dig
    assert An.PROTOCOL_EXPECTED_TEMPERATURE == {t: C.PROTOCOL_SAMPLING_BY_TASK[t].executed.temperature for t in An.TASKS}
    assert An.remote_ladder_dir("s", model_slug="nemotron3nano4b") == An._volume_relative(RP.ladder_dir(C.VOL_GEN_ROOT, "nemotron3nano4b", "s"))
    with pytest.raises(An.AnalysisError, match="needs --out-suffix"):
        An.remote_ladder_dir("", model_slug="nemotron3nano4b")


def test_store_object_sha256_is_the_ladders_reader_and_refuses_two_objects():
    m = {C.MANIFEST_KEY_ROUTE: {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "a" * 64}, C.MANIFEST_KEY_PROVENANCE: {"artifact_sha256": "a" * 64}}
    assert An.store_object_sha256(m, cell="x") == ("a" * 64, f"{C.MANIFEST_KEY_ROUTE}.{C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256}")
    m[C.MANIFEST_KEY_PROVENANCE]["artifact_sha256"] = "b" * 64
    with pytest.raises(An.AnalysisError, match="two objects claimed for one store"):
        An.store_object_sha256(m, cell="x")
    assert An.store_object_sha256({}, cell="x") == (None, None)


def test_lib_families_go_through_the_contracts_reader(tmp_path):
    repo = F.make_pins(tmp_path / "repo")
    assert An.load_lib_families(repo) == C.bcb_lib_families(repo)
    p = Path(repo) / An.LIB_FAMILIES_REL
    doc = json.loads(p.read_text()); doc["families"] = {}
    p.write_text(json.dumps(doc))
    with pytest.raises(An.AnalysisError, match="families must be a non-empty"):
        An.load_lib_families(repo)


# --------------------------------------------------------------------------- #
# the generation guard                                                        #
# --------------------------------------------------------------------------- #


def test_generation_guard_refuses_a_relabelled_sidecar(world):
    p = world["it"] / CONST_LORA / An.LANE_ROSTER_SIDECAR
    doc = json.loads(p.read_text()); doc["object_generation"] = "final"; doc["superseded_by_key"] = None       # relabelled, digest stale
    p.write_text(json.dumps(doc))
    with pytest.raises(An.AnalysisError, match="does not recompute"):
        _analyze(world)


def test_generation_guard_rederives_the_label_from_the_roster_at_the_producer_commit(world):
    """A relabelled sidecar whose digest WAS recomputed passes the self-contained checks
    and the served-object binding — and still refuses, because the label is re-derived from the roster the store's
    PRODUCER COMMIT carried (`git show <producer.tree_sha>:roster.yaml`, hashed to the sidecar's roster sha). A roster
    that merely EXISTS — on a foreign branch, in the working file — proves nothing: a sidecar regenerated from a
    relabelled roster committed on another branch (same object digest, generation flipped to `final`) refuses, while
    the legitimate historical roster at the producer commit passes after the file moved on."""
    def relabel(b):
        b["object_generation"], b["superseded_by_key"] = "final", None
    write_sidecar(world["it"] / CONST_LORA, roster=world["doc"], roster_path=world["roster"], wave="gemma4e4b_it", entry_id="const_lora", mutate=relabel)
    with pytest.raises(An.AnalysisError, match="disagrees with the roster at the producer commit .*object_generation: sidecar 'final' vs roster 'batch8-provisional'"):
        _analyze(world)

    def unknown(b):
        b["roster_sha256"] = "9" * 64
    write_sidecar(world["it"] / CONST_LORA, roster=world["doc"], roster_path=world["roster"], wave="gemma4e4b_it", entry_id="const_lora", mutate=unknown)
    with pytest.raises(An.AnalysisError, match=f"roster at the store's producer commit {TREE[:12]}.* hashes to .*, the sidecar names 999999999999"):
        _analyze(world)
    # a FOREIGN BRANCH commits a relabelled roster (object digest unchanged, generation `final`); a sidecar regenerated from it
    # is internally perfect and its roster sha IS a committed revision — but not the producer commit's: refused
    lane = world["roster"].parent
    _git("checkout", "-q", "-b", "foreign", cwd=lane)
    relabelled = fixture_roster()
    entry = relabelled["waves"]["gemma4e4b_it"]["entries"]["const_lora"]
    entry["object"]["object_generation"] = "final"
    entry["object"].pop("superseded_by_key", None)
    entry.pop("rerun_trigger", None)
    write_roster(world["roster"], relabelled)
    foreign = commit_roster(world["roster"], message="relabel on a foreign branch")
    assert foreign != TREE
    write_sidecar(world["it"] / CONST_LORA, roster=relabelled, roster_path=world["roster"], wave="gemma4e4b_it", entry_id="const_lora")
    _git("checkout", "-q", "-", cwd=lane)                          # back to the producer's branch: the working file is the original again
    assert W.roster_sha256(world["roster"]) != json.loads((world["it"] / CONST_LORA / An.LANE_ROSTER_SIDECAR).read_text())["roster_sha256"]
    with pytest.raises(An.AnalysisError, match="hashes to .*, the sidecar names .*regenerated from another roster"):
        _analyze(world)
    # the legitimate sidecar (the producer commit's roster) passes even after the working file moved on to a NEW commit on this branch
    write_sidecar(world["it"] / CONST_LORA, roster=world["doc"], roster_path=world["roster"], wave="gemma4e4b_it", entry_id="const_lora")
    moved = fixture_roster(); moved["waves"]["gemma4e4b_it"]["entries"]["none"]["note"] = "moved on after the launch"
    write_roster(world["roster"], moved)
    assert commit_roster(world["roster"], message="moved on") not in (TREE, foreign)
    a = _analyze(world)
    assert a["generations"][CONST_LORA]["object_generation"] == "batch8-provisional" and a["generations"][CONST_LORA]["roster_moved_since_launch"] is True
    assert a["generations"][CONST_LORA]["roster_at_launch_source"] == f"git {TREE[:12]}:roster.yaml"
    # the resolver itself: an unknown producer commit, a roster outside any repository, a malformed tree sha all refuse
    with pytest.raises(An.AnalysisError, match="producer commit eeeeeeeeeeee.* is unknown"):
        An.roster_at_producer_commit(world["roster"], "e" * 40, "9" * 64)
    loose = write_roster(world["tmp"] / "loose" / "roster.yaml")
    with pytest.raises(An.AnalysisError, match="is not in a git repository"):
        An.roster_at_producer_commit(loose, TREE, W.roster_sha256(loose))
    with pytest.raises(An.AnalysisError, match="is not a 40-hex commit"):
        An.roster_at_producer_commit(world["roster"], "5039ce9", W.roster_sha256(loose))


def test_generation_guard_refuses_a_sidecar_whose_object_disagrees_with_the_store(world):
    def wrong_object(b):
        b["object_sha256"] = "f" * 64
    write_sidecar(world["it"] / CONST_PSR, roster=world["doc"], roster_path=world["roster"], wave="gemma4e4b_it", entry_id="const_psr", mutate=wrong_object)
    with pytest.raises(An.AnalysisError, match="object_sha256 .* != the object the store served"):                # re-digested, wrong object
        _analyze(world)


def test_generation_guard_refuses_a_fitted_store_without_a_sidecar_and_a_snapshot_of_another_launch(world):
    (world["it"] / CONST_LORA / An.LANE_ROSTER_SIDECAR).unlink()
    with pytest.raises(An.AnalysisError, match="has no lane_roster.json"):
        _analyze(world)

    def other_wave(b):
        b["wave"] = "gemma4e4b_main"
    write_sidecar(world["it"] / CONST_LORA, roster=world["doc"], roster_path=world["roster"], wave="gemma4e4b_it", entry_id="const_lora", mutate=other_wave)
    with pytest.raises(An.AnalysisError, match="wave 'gemma4e4b_main' != 'gemma4e4b_it'"):
        _analyze(world)


def test_generation_guard_none_needs_no_sidecar_and_two_fitted_generations_refuse_pairing():
    base = {"cell": NONE, "fitted": False, "object_generation": None}
    b8 = {"cell": CONST_LORA, "fitted": True, "object_generation": "batch8-provisional"}
    final = {"cell": DIFF_LORA, "fitted": True, "object_generation": "final"}
    An.check_pairing_generations(base, b8)                       # batch-8 vs its own none: fine
    An.check_pairing_generations(base, final)
    with pytest.raises(An.AnalysisError, match="different object generations"):
        An.check_pairing_generations(b8, final)


def test_generation_label_is_the_sidecars_not_the_current_rosters(world, monkeypatch):
    # the CURRENT roster flips the object to the eb64 re-fit (a new sha, generation final); the OLD store keeps its label,
    # RE-DERIVED from the launched roster, which git still holds at the sidecar's sha
    doc = fixture_roster()
    doc["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["object"] = _obj(CONST_LORA, gen="final", staged="eb64__" + CONST_LORA, sha="e" * 64)
    write_roster(world["roster"], doc)
    commit_roster(world["roster"])
    with pytest.raises(An.AnalysisError, match="fails widen_roster.validate"):        # the sha is off the pins: Slice B's validator refuses first
        _analyze(world)
    # moving the pins WITH the roster: the validator then passes and the store's label is still the sidecar's
    monkeypatch.setattr(An, "validate_roster", lambda roster, roster_path: {"roster_validated_by": None, "note": "pins moved with the roster (test)"})
    a = _analyze(world)
    g = a["generations"][CONST_LORA]
    assert g["object_generation"] == "batch8-provisional" and g["provisional"] and g["roster_object_drift"] is True and g["roster_moved_since_launch"] is True
    assert g["roster_at_launch_source"] == f"git {TREE[:12]}:roster.yaml" and g["roster_sha256_at_launch"] != W.roster_sha256(world["roster"])
    assert a["legs"][BCB]["paired"][CONST_LORA]["provisional"] is True
    # a launch from an UNCOMMITTED roster (the working file edited past the producer commit) is unverifiable: refuse
    doc_a = fixture_roster(); doc_a["waves"]["gemma4e4b_it"]["entries"]["none"]["note"] = "uncommitted state A"
    write_roster(world["roster"], doc_a)                                # never committed; the stores still name the producer commit
    launched_from_a = make_wave(world["tmp"] / f"{SUFFIX_IT}-uncommitted", doc_a, "gemma4e4b_it", roster_path=world["roster"], run_id=RUN_IT)
    with pytest.raises(An.AnalysisError, match="hashes to .*, the sidecar names .*regenerated from another roster"):
        _analyze(world, rung1=launched_from_a)


def test_a_root_with_a_sampling_validation_failure_is_not_protocol_evidence(world):
    """The worker files `sampling_validation_failed.json` at a configuration's out root when
    the clamp override did not hold (BFCL executed at 0.01 behind the protocol label). The analysis refuses that root
    by name, whatever its DONE markers say — beside a fitted store's lane_roster.json and on the object-free `none` alike;
    an unreadable marker is a failure too; the same store without the marker admits."""
    assert _analyze(world)["cells"] == W.runnable_cells(world["doc"], "gemma4e4b_it")                 # without the marker: admitted
    assert (world["it"] / CONST_LORA / An.LANE_ROSTER_SIDECAR).is_file()                    # the marker sits beside the sidecar
    p = F.write_sampling_validation_marker(world["it"] / CONST_LORA, cell=CONST_LORA, run_id=RUN_IT, fcid=f"fc-{CONST_LORA[:6]}", tree_sha=TREE)
    with pytest.raises(An.AnalysisError, match=f"carries {C.SAMPLING_VALIDATION_FAILED_FILENAME} \\({C.REC_CLAMP_OVERRIDE_INEFFECTIVE}: .*not protocol evidence"):
        _analyze(world)
    p.unlink()
    p_none = world["it"] / NONE / C.SAMPLING_VALIDATION_FAILED_FILENAME
    p_none.write_text("{not json")
    with pytest.raises(An.AnalysisError, match=f"\\({C.REC_SAMPLING_VALIDATION_FAILED}: unreadable.*not protocol evidence"):
        _analyze(world)
    p_none.unlink()
    assert _analyze(world)["cells"] == W.runnable_cells(world["doc"], "gemma4e4b_it")


# --------------------------------------------------------------------------- #
# same-wave provenance: the suffix is the durable identity                      #
# --------------------------------------------------------------------------- #


def test_a_resumed_cell_with_a_second_attempt_is_one_wave_when_bound_to_the_suffix(world):
    """A resumed configuration mints a new run id and appends a second attempt pointer under the SAME suffix:
    accepted, every attempt listed, `run_id` unset (two invocations), the sweep of each attempt required."""
    second = f"{SHA7}-99999999"
    fcid = sandbox_and_pointer(world["it"] / CONST_PSR, run_id=second, cell=CONST_PSR, suffix=SUFFIX_IT, model_slug="gemma4e4b-f22it",
                               fcid="fc-resume-1", started_at=T0 + 7200)
    a = _analyze(world)
    prov = a["identity"]["provenance"]
    assert prov["run_ids"] == sorted([RUN_IT, second]) and prov["run_id"] is None and prov["n_attempts"] == len(a["cells"]) + 1 and prov["one_wave"] is True
    assert [r["function_call_id"] for r in prov["attempts_by_cell"][CONST_PSR]] == [f"fc-{CONST_PSR[:6]}", fcid]
    rows = {(r["cell_id"], r["eval_id"]): r for r in An.paper_rows(a) if r["status"] == "graded"}
    assert f"run_ids={RUN_IT},{second}" in rows[(CONST_PSR, "bfcl_200")]["provenance"]
    md = An.render_wave_doc(a)
    assert f"`{RUN_IT}`, `{second}` ({len(a['cells']) + 1} attempt(s))" in md


def test_a_failed_standup_attempt_is_admitted_on_its_watcher_sweep_and_the_retry_completes_the_wave(world, tmp_path):
    """A BCB configuration whose vLLM standup fails before the driver starts leaves an attempt
    pointer (published before the standup) with an EMPTY event record (the launcher creates it before publishing the
    pointer) and the WATCHER's sweep — verified empty, nothing listed; a later attempt under the same suffix completes.
    The wave passes provenance with every attempt listed and the failed one marked `driver_started` False. The LEGACY
    shape (no record file at all) is admitted ONLY on that same authoritative sweep — zero listed, zero terminated —
    and refused as before otherwise; an EMPTY record beside a sweep that found sandboxes is a torn record."""
    state = tmp_path / "sandbox-state"
    failed_run, retry_run = f"{SHA7}-aaaaaaaa", f"{SHA7}-bbbbbbbb"
    # the new producer shape: pointer + EMPTY sandboxes_<fcid>.jsonl + the watcher's verified-empty sweep
    sandbox_and_pointer(world["it"] / CONST_PSR, run_id=failed_run, cell=CONST_PSR, suffix=SUFFIX_IT, model_slug="gemma4e4b-f22it",
                        fcid="fc-standup-fail", n=0, sweep=False, started_at=T0 + 3600)
    assert (world["it"] / CONST_PSR / "watch" / "sandboxes_fc-standup-fail.jsonl").stat().st_size == 0
    watcher_sweep(state, run_id=failed_run, cell=CONST_PSR, fcid="fc-standup-fail")
    sandbox_and_pointer(world["it"] / CONST_PSR, run_id=retry_run, cell=CONST_PSR, suffix=SUFFIX_IT, model_slug="gemma4e4b-f22it",
                        fcid="fc-retry", started_at=T0 + 7200)
    a = _analyze(world)
    prov = a["identity"]["provenance"]
    assert prov["run_ids"] == sorted([RUN_IT, failed_run, retry_run]) and prov["run_id"] is None and prov["n_attempts"] == len(a["cells"]) + 2
    by_fcid = {r["function_call_id"]: r for r in prov["attempts_by_cell"][CONST_PSR]}
    assert set(by_fcid) == {f"fc-{CONST_PSR[:6]}", "fc-standup-fail", "fc-retry"}
    failed = by_fcid["fc-standup-fail"]
    assert failed["driver_started"] is False and failed["n_events"] == 0 and failed["legacy_no_record"] is False
    assert failed["events_record"].endswith("sandboxes_fc-standup-fail.jsonl") and failed["sweep_certificate"].startswith(str(state))
    assert by_fcid["fc-retry"]["driver_started"] is True and by_fcid["fc-retry"]["n_events"] == 11
    assert a["cost"]["sandboxes"][CONST_PSR]["n_created"] == 12                # the retry's six + the original's six; the failed attempt none
    # the LEGACY shape: no record file at all — admitted ONLY on the watcher's zero-listed, zero-terminated verified-empty sweep
    (world["it"] / CONST_PSR / "watch" / "sandboxes_fc-standup-fail.jsonl").unlink()
    a = _analyze(world)
    legacy = {r["function_call_id"]: r for r in a["identity"]["provenance"]["attempts_by_cell"][CONST_PSR]}["fc-standup-fail"]
    assert legacy["legacy_no_record"] is True and legacy["driver_started"] is False and legacy["events_record"] is None
    watcher_sweep(state, run_id=failed_run, cell=CONST_PSR, fcid="fc-standup-fail", listed=1)                     # the sweep saw a sandbox
    with pytest.raises(An.AnalysisError, match="has no watch/sandboxes_fc-standup-fail.jsonl record .*does not prove the driver never started: listed 1"):
        _analyze(world)
    watcher_sweep(state, run_id=failed_run, cell=CONST_PSR, fcid="fc-standup-fail", terminated=["sb-ghost"])         # it terminated one
    with pytest.raises(An.AnalysisError, match="has no watch/sandboxes_fc-standup-fail.jsonl record .*terminated \\['sb-ghost'\\]"):
        _analyze(world)
    (state / failed_run / "sweep_fc-standup-fail.json").unlink()                                                    # no sweep at all
    with pytest.raises(An.AnalysisError, match="has no watch/sandboxes_fc-standup-fail.jsonl record; .*is not covered — no sweep certificate"):
        _analyze(world)
    # an EMPTY record beside a sweep that found sandboxes is a torn record, refused
    (world["it"] / CONST_PSR / "watch" / "sandboxes_fc-standup-fail.jsonl").write_text("")
    watcher_sweep(state, run_id=failed_run, cell=CONST_PSR, fcid="fc-standup-fail", terminated=["sb-ghost"])
    with pytest.raises(An.AnalysisError, match="EMPTY watch/sandboxes_fc-standup-fail.jsonl record but its sweep found sandboxes"):
        _analyze(world)


@pytest.mark.parametrize("tweak,needle", [
    (dict(run_id="deadbee-00000001"), "minted at sha deadbee, not this suffix's"),
    (dict(run_id="not-a-run-id"), "is not the launcher's <sha7>-<hex8>"),
    (dict(suffix=f"bcbw-gemma4e4b_it-{SHA7}-smoke"), "a pointer from another suffix"),
    (dict(model_slug="gemma4e4b-f22"), "a pointer from another suffix"),
    (dict(tree_sha="e" * 40), "!= the stores' producer tree"),
    (dict(sweep=False), "no sweep certificate"),
])
def test_provenance_refuses_an_attempt_from_another_suffix_sha_or_without_its_sweep(world, tweak, needle):
    kw = dict(run_id=f"{SHA7}-99999999", cell=CONST_PSR, suffix=SUFFIX_IT, model_slug="gemma4e4b-f22it", fcid="fc-resume-2")
    kw.update(tweak)
    sandbox_and_pointer(world["it"] / CONST_PSR, **kw)
    with pytest.raises(An.AnalysisError, match=needle):
        _analyze(world)


def test_provenance_refuses_a_missing_pointer_a_foreign_suffix_dir_and_accepts_the_watchers_sweep(world, tmp_path):
    for pattern in ("attempt_*.json", "sweep_*.json"):
        for f in (world["it"] / PROMPTED / "watch").glob(pattern):
            f.unlink()
    with pytest.raises(An.AnalysisError, match="no watch/attempt_\\*.json pointer"):
        _analyze(world)
    # the external watcher's sweep certificate (state dir) covers an attempt whose container left none
    sandbox_and_pointer(world["it"] / PROMPTED, run_id=RUN_IT, cell=PROMPTED, suffix=SUFFIX_IT, model_slug="gemma4e4b-f22it", sweep=False)
    with pytest.raises(An.AnalysisError, match="no sweep certificate"):
        _analyze(world)
    state = tmp_path / "sandbox-state" / RUN_IT
    state.mkdir(parents=True)
    owner = {"run_id": RUN_IT, "cell": PROMPTED, "function_call_id": f"fc-{PROMPTED[:6]}"}
    (state / f"sweep_fc-{PROMPTED[:6]}.json").write_text(json.dumps({"swept_at": T0, "by": "watcher", "owner": owner, "ids_terminated": [], "left": [],
                                                                    "could_not_terminate": [], "still_live": [], "tags_unreadable": [], "verified_empty": True,
                                                                    "rounds": 1, "n_listed_first_round": 0}))
    a = _analyze(world)
    assert a["identity"]["provenance"]["attempts_by_cell"][PROMPTED][0]["sweep_certificate"].startswith(str(state))
    # a sweep that did not verify empty does not cover the attempt
    (state / f"sweep_fc-{PROMPTED[:6]}.json").write_text(json.dumps({"owner": owner, "verified_empty": False, "still_live": ["sb-1"], "could_not_terminate": []}))
    with pytest.raises(An.AnalysisError, match="did not verify empty"):
        _analyze(world)
    # the pulled dir's name IS the suffix: a dir under another grammar, or another wave's suffix, refuses
    other = world["tmp"] / "renamed-dir"
    world["main"].rename(other)
    with pytest.raises(An.AnalysisError, match="is not the lane's `bcbw-<wave>-<sha7>"):
        _analyze(world, wave="gemma4e4b_main", rung1=other)
    other.rename(world["tmp"] / f"bcbw-gemma4e4b_it-{SHA7}-x")
    with pytest.raises(An.AnalysisError, match="names wave 'gemma4e4b_it', not the analysed 'gemma4e4b_main'"):
        _analyze(world, wave="gemma4e4b_main", rung1=world["tmp"] / f"bcbw-gemma4e4b_it-{SHA7}-x")


# --------------------------------------------------------------------------- #
# the protocol record, the roster's gate, the fit layers                       #
# --------------------------------------------------------------------------- #


def test_protocol_record_refusals(world, tmp_path):
    doc, rp = world["doc"], world["roster"]
    clamped = make_wave(tmp_path / f"{SUFFIX_MAIN}-clamped", doc, "gemma4e4b_main", roster_path=rp, bfcl_executed=0.01)
    with pytest.raises(An.AnalysisError, match="executed_temperature FAILS"):
        _analyze(world, wave="gemma4e4b_main", rung1=clamped)
    no_override = make_wave(tmp_path / f"{SUFFIX_MAIN}-noov", doc, "gemma4e4b_main", roster_path=rp, clamp=False)
    with pytest.raises(An.AnalysisError, match="sampling_clamp_override.*absent under protocol"):          # the wave-wide identity refuses first
        _analyze(world, wave="gemma4e4b_main", rung1=no_override)
    # the protocol record itself files ONE problem for the absent override (not a from/to mismatch against None as well)
    spec = An.wave_spec(doc, "gemma4e4b_main")
    stores = {(c, t): L.load_store(no_override, c, t) for c in spec.cells for t in spec.tasks}
    with pytest.raises(An.AnalysisError, match="sampling_clamp_override absent on a protocol store") as ei:
        An.protocol_section(stores, cells=spec.cells, tasks=spec.tasks, wave=spec, mode="protocol")
    assert str(ei.value).count("sampling_clamp_override") == 1, str(ei.value)
    card = make_wave(tmp_path / f"{SUFFIX_MAIN}-card", doc, "gemma4e4b_main", roster_path=rp, protocol=False)
    with pytest.raises(An.AnalysisError, match="roster declares 'protocol'"):
        _analyze(world, wave="gemma4e4b_main", rung1=card)
    # the legacy card decode string names the model card and never the word "protocol"
    spec = An.wave_spec(doc, "gemma4e4b_main")
    stores = {(c, t): L.load_store(card, c, t) for c in spec.cells for t in spec.tasks}
    card_spec = An.WaveSpec(**{**spec.__dict__, "sampling": "card", "sampling_clamp_override": None})
    pr = An.protocol_section(stores, cells=spec.cells, tasks=spec.tasks, wave=card_spec, mode="card")
    assert all(d.startswith("model card decoding") and "protocol" not in d for d in pr["decode_by_task"].values())
    # a protocol / card MIX across the wave's stores is not one experiment
    _edit_manifest(world["main"], DIFF_PSR, BCB, lambda m: (m[C.MANIFEST_KEY_SAMPLING].__setitem__("mode", "card"),        # one store files both legs
                                                            m[C.MANIFEST_KEY_SAMPLING].pop("by_task"), m[C.MANIFEST_KEY_TINYCAP].pop("sampling_by_task")))
    with pytest.raises(An.AnalysisError, match="several sampling modes|sampling.mode"):
        _analyze(world, wave="gemma4e4b_main", rung1=world["main"])


def test_roster_gates_the_stores(world, tmp_path):
    doc, rp = world["doc"], world["roster"]
    # an extra store for a cell the roster lists as QUEUED refuses
    extra = make_wave(tmp_path / f"{SUFFIX_IT}-extra", doc, "gemma4e4b_it", roster_path=rp)
    F.make_multi_store(extra, "constv2ctl2-full__lora_r4_ce", {t: F.rung1_results("constv2ctl2-full__lora_r4_ce", t) for t in (BCB, BFCL)},
                       caps=dict(F.CAP_OFF), served=F.SERVED_RUNG1, limits={}, object_sha256="c" * 64)
    with pytest.raises(An.AnalysisError, match="does not list as runnable"):
        _analyze(world, rung1=extra)
    # a missing runnable cell refuses
    import shutil
    shutil.rmtree(world["it"] / CONST_PSR)
    with pytest.raises(An.AnalysisError, match="have no store here"):
        _analyze(world)
    # a wave whose store files a BLOCKED leg refuses
    qwen_full = make_wave(tmp_path / f"{SUFFIX_QWEN}-full", fixture_roster(bfcl_qwen="runnable"), "qwen35_4b_main", roster_path=rp)
    with pytest.raises(An.AnalysisError, match="legs the roster marks blocked"):
        _analyze(world, wave="qwen35_4b_main", rung1=qwen_full)


def test_qwen_wave_bcb_only_with_bfcl_blocked_and_qwen_fit_layers(world, tmp_path):
    rung1 = make_wave(tmp_path / SUFFIX_QWEN, world["doc"], "qwen35_4b_main", roster_path=world["roster"], run_id=RUN_QWEN)
    a = _analyze(world, wave="qwen35_4b_main", rung1=rung1)
    assert a["tasks"] == [BCB] and list(a["blocked_legs"]) == [BFCL] and set(a["breakdowns"]) == {"bcb_by_family"} and set(a["quoted"]) == {"bcb_failures"}
    assert a["identity"]["model_id"] == QWEN_HF and a["identity"]["common"]["model_row"]["hf"] == QWEN_HF
    fl = a["fit_layers"][DIFF_LORA]
    assert fl["fit_layers"] == "3,7,11,15,19,23,27,31" and fl["fit_layer_count"] == 8 and fl["layer_key"] == "layers"
    rows = An.paper_rows(a)
    blocked = [r for r in rows if r["eval_id"] == "bfcl_200"]
    # since 2026-09-22 the qwen BFCL leg is QUEUED on the LIVE half of the S5 probe; a
    # not-run row's status is `<leg status>: <reason>` verbatim from the roster, whatever the status word
    assert len(blocked) == 2 and all(r["status"].startswith("queued: S5 half-verified") and r["value"] is None for r in blocked)
    # a not-run row still carries the DECLARED protocol (labelled) so the notebook's protocol selection keeps it in view, and no object
    assert all(r["decode"].startswith("benchmark protocol (declared by the roster; NOT RUN): BFCL-200 T=0.001") for r in blocked)
    assert all(r["object_artifact_sha256"] is None and r["thinking"] == "off" for r in blocked)
    assert {r["view"] for r in rows} == {"qwen35_4b"} and {r["lane_kind"] for r in rows} == {"base→instruct"} and {r["model"] for r in rows} == {"qwen35_4b"}
    An.make_figures(a, tmp_path / "figs")            # the BFCL breakdown renders its 'not applicable' placeholder


def test_fit_layers_come_from_the_sidecar_and_are_cross_checked(world):
    it = world["it"]
    entry = An.wave_spec(world["doc"], "gemma4e4b_it").entry_of_cell(CONST_LORA)
    (it / An.OBJECTS_SUBDIR / CONST_LORA / "train_budget_meta.json").unlink()
    with pytest.raises(An.AnalysisError, match="is missing — the fit layers are read per cell"):
        An.fit_layers_of(it, entry, served_sha=SHA[CONST_LORA])
    write_object_sidecar(it, CONST_LORA, filename="train_budget_meta.json", layers=list(range(20)))
    with pytest.raises(An.AnalysisError, match="!= the roster's declaration 0-41"):
        An.fit_layers_of(it, entry, served_sha=SHA[CONST_LORA])
    write_object_sidecar(it, CONST_LORA, filename="train_budget_meta.json", layers=list(range(42)))
    assert An.fit_layers_of(it, entry, served_sha=SHA[CONST_LORA])["fit_layer_count"] == 42
    qentry = An.wave_spec(world["doc"], "qwen35_4b_main").entry_of_cell(DIFF_LORA)
    write_object_sidecar(it, DIFF_LORA, filename="derived_meta.json", layers=[3, 7, 11, 15, 19, 23, 27, 31], object_sha="0" * 64, derived=True)
    with pytest.raises(An.AnalysisError, match="sidecar's object_sha256 .* != the object the store served"):
        An.fit_layers_of(it, qentry, served_sha=SHA["qwen_diff"])


def test_projection_consumer_on_full_to_single_leg_stores(world, tmp_path):
    """A full (both-legs) store and a BCB-only store of the same wave compare equal on the RETAINED leg —
    the per-task maps projected onto BCB, the joined `sampling.source` excluded; a changed executed temperature on
    the retained task refuses; a protocol / card mix refuses."""
    spec = An.wave_spec(world["doc"], "gemma4e4b_main")
    full = L.load_store(world["main"], NONE, BCB)
    one = F.make_multi_store(tmp_path / "one", DIFF_LORA, {BCB: F.rung1_results(DIFF_LORA, BCB)}, caps={BCB: 4096}, served=F.SERVED_RUNG1, limits={},
                             object_sha256=SHA[DIFF_LORA], tree_sha=TREE)
    protocolize_store(one, tasks=[BCB], served_sha=SHA[DIFF_LORA], cell=DIFF_LORA, kind="lora_subtraction", arm="x")
    st = L.load_store(tmp_path / "one", DIFF_LORA, BCB)
    stores = {(NONE, BCB): full, (DIFF_LORA, BCB): st}
    common = An.common_identity_check(stores, cells=[NONE, DIFF_LORA], tasks=[BCB], wave=spec, mode="protocol")
    assert common["one_experiment"] is True, common
    chk = An.identity_check({NONE: full.manifest, DIFF_LORA: st.manifest}, task=BCB, pin=F.BCB_PIN, mode="protocol")
    assert chk["same_instrument"] is True, chk["mismatched_required"]
    # a changed executed temperature on the retained task
    m = json.loads((one / "manifest.json").read_text()); m[C.MANIFEST_KEY_TINYCAP]["sampling_by_task"][BCB]["temperature"] = 0.7
    m[C.MANIFEST_KEY_SAMPLING]["by_task"][BCB]["executed"]["temperature"] = 0.7
    (one / "manifest.json").write_text(json.dumps(m))
    st2 = L.load_store(tmp_path / "one", DIFF_LORA, BCB)
    chk = An.identity_check({NONE: full.manifest, DIFF_LORA: st2.manifest}, task=BCB, pin=F.BCB_PIN, mode="protocol")
    assert chk["same_instrument"] is False and any("sampling_by_task" in k or "by_task" in k for k in chk["mismatched_required"])
    # the refusal carries the DIFFERING VALUES of a projected task map (dug from `maps`, not `scalars`)
    rec = next(v for k, v in chk["mismatched_required"].items() if k.endswith(f"tinycap.sampling_by_task[{BCB}]"))
    assert rec["base"]["temperature"] == 0.0 and rec["cell"]["temperature"] == 0.7
    rec = next(v for k, v in chk["mismatched_required"].items() if k.endswith(f"sampling.by_task[{BCB}]"))
    assert rec["base"]["executed"]["temperature"] == 0.0 and rec["cell"]["executed"]["temperature"] == 0.7
    # a card store beside a protocol one
    m = json.loads((one / "manifest.json").read_text()); m[C.MANIFEST_KEY_TINYCAP].pop("sampling_by_task"); m[C.MANIFEST_KEY_SAMPLING].pop("by_task")
    m[C.MANIFEST_KEY_SAMPLING]["mode"] = "card"
    (one / "manifest.json").write_text(json.dumps(m))
    st3 = L.load_store(tmp_path / "one", DIFF_LORA, BCB)
    common = An.common_identity_check({(NONE, BCB): full, (DIFF_LORA, BCB): st3}, cells=[NONE, DIFF_LORA], tasks=[BCB], wave=spec, mode="protocol")
    assert common["one_experiment"] is False and "sampling.mode" in common["mismatched"]


# --------------------------------------------------------------------------- #
# paper rows, summary, doc, CLI                                                #
# --------------------------------------------------------------------------- #


def test_paper_rows_carry_the_lane_view_marks_and_queued_rows_and_append(world, tmp_path):
    import pandas as pd
    a = _analyze(world)
    rows = An.paper_rows(a)
    assert all(tuple(r) == An.PAPER_COLUMNS for r in rows) and len(An.PAPER_COLUMNS) == 48
    assert An.PAPER_COLUMNS.index("object_artifact_sha256") == An.PAPER_COLUMNS.index("fit_layer_count") + 1 and "thinking" in An.PAPER_COLUMNS
    # EVERY row's decode carries the literal word "protocol"; `thinking` is "off" on every row
    assert all("protocol" in r["decode"] and r["decode"].startswith("benchmark protocol") for r in rows) and {r["thinking"] for r in rows} == {"off"}
    measured = [r for r in rows if r["status"] in ("graded", "pending ladder")]
    queued = [r for r in rows if r["status"].startswith("queued")]
    n_queued = sum(1 for e in a["wave"]["entries"] if e["status"] == "queued")
    n_not_run = sum(1 for e in a["wave"]["entries"] if e["status"] != "runnable")
    assert len(measured) == len(a["cells"]) * len(a["tasks"]) and len(queued) == n_queued * len(An.TASKS) and all(r["source"] == "bcb_bfcl_200_widen" for r in rows)
    assert len(rows) == expected_paper_rows(a) == len(measured) + n_not_run * len(An.TASKS)
    si = next(r for r in measured if r["cell_id"] == CONST_SI and r["eval_id"] == "bfcl_200")
    assert (si["arm"], si["family"], si["object_type"], si["object_artifact_sha256"], si["fit_layers"]) == ("si_constitution", "prompted", "prompt", None, None)
    by = {(r["cell_id"], r["eval_id"]): r for r in measured}
    for cell in a["cells"]:
        for ev in ("bigcodebench_200", "bfcl_200"):
            r = by[(cell, ev)]
            assert (r["model"], r["lane_kind"], r["view"], r["dose"]) == ("gemma4e4b", "instruct-direct", "gemma4e4b_it", 1.0)
    # BCB fired and is not merged: pending, no reading (rung 1 named as diagnostic only); BFCL graded with the protocol decode string
    pend = by[(CONST_LORA, "bigcodebench_200")]
    assert pend["status"] == "pending ladder" and pend["value"] is None and pend["delta"] is None and "diagnostic only" in pend["note"]
    g = by[(CONST_LORA, "bfcl_200")]
    assert g["status"] == "graded" and g["arm"] == "const_lora" and g["family"] == "constitution" and g["object_type"] == "lora"
    assert g["decode"].startswith("benchmark protocol: BFCL-200 T=0.001 (declared 0.001; equal (clamp override applied)), top_p 1.0, top_k 0, thinking OFF, k=1")
    assert "clamp override 0.01->0.001" in g["decode"] and g["decode_cap_tokens"] == 1024
    assert "object_generation=batch8-provisional" in g["note"] and "superseded_by_key=gemma4e4b_it_doc_eb64" in g["provenance"] and "do not pool" in g["note"]
    assert (g["fit_layers"], g["fit_layer_count"]) == ("0-41", 42) and g["delta"] is not None and "final reading (rung1):" in g["delta_method"]
    # the served object's sha rides the row (`object_artifact_sha256`); null on the object-free cells
    assert g["object_artifact_sha256"] == SHA[CONST_LORA] and by[(CONST_PSR, "bfcl_200")]["object_artifact_sha256"] == SHA[CONST_PSR]
    assert f"suffix={SUFFIX_IT}; run_ids={RUN_IT}" in g["provenance"]
    p = by[(PROMPTED, "bfcl_200")]
    assert (p["arm"], p["family"], p["object_type"]) == ("si_safety", "prompted", "prompt") and p["fit_layers"] is None and "provisional" not in p["note"]
    assert p["object_artifact_sha256"] is None
    n = by[(NONE, "bfcl_200")]
    assert n["delta"] is None and n["fit_layers"] is None and n["arm"] == "none" and n["object_artifact_sha256"] is None
    q = {r["arm"]: r for r in queued if r["eval_id"] == "bfcl_200"}
    assert q["alphasteer_as_fit"]["cell_id"] is None and q["alphasteer_as_fit"]["object_type"] == "matrix" and q["alphasteer_as_fit"]["value"] is None
    assert q["ctl_lora"]["status"].startswith("queued: no instruct-direct fit") and q["ctl_lora"]["note"].startswith("unblock:")
    assert q["ctl_lora"]["decode"] == "benchmark protocol (declared by the roster; NOT RUN): BFCL-200 T=0.001, top_p 1.0, top_k 0, thinking OFF, k=1; vLLM clamp override 0.01->0.001 (D13)"
    assert q["ctl_lora"]["object_artifact_sha256"] is None and q["ctl_lora"]["decode_cap_tokens"] is None
    # append semantics: wave A, then wave B, then A again replaces A's rows only
    pq = tmp_path / "rows.parquet"
    n_a = len(rows)
    rec = An.write_paper_rows(a, pq)
    assert rec["n_added"] == n_a and rec["n_replaced"] == 0 and rec["n_rows_written"] == n_a
    b = _analyze(world, wave="gemma4e4b_main", rung1=world["main"])
    n_b = len(An.paper_rows(b))
    assert n_b == expected_paper_rows(b)
    rec = An.write_paper_rows(b, pq)
    assert rec["n_added"] == n_b and rec["n_replaced"] == 0 and rec["n_kept"] == n_a and rec["n_rows_written"] == n_a + n_b
    rec = An.write_paper_rows(a, pq)
    assert rec["n_added"] == n_a and rec["n_replaced"] == n_a and rec["n_kept"] == n_b and rec["n_rows_written"] == n_a + n_b
    df = pd.read_parquet(pq)
    assert list(df.columns) == list(An.PAPER_COLUMNS) and str(df["k"].dtype) == "Int64" and str(df["saturated"].dtype) == "bool"
    assert set(df["view"]) == {"gemma4e4b_it", "gemma4e4b"} and set(df["lane_kind"]) == {"instruct-direct", "base→instruct"}
    main_rows = df[df.view == "gemma4e4b"]
    assert set(main_rows.arm) == {"none", "diff_lora", "diff_psr"} and "WEAK gemma residual" in main_rows[main_rows.arm == "diff_psr"].note.iloc[0]
    assert set(main_rows.object_artifact_sha256.dropna()) == {SHA[DIFF_LORA], SHA[DIFF_PSR]}
    assert An.write_paper_rows(a, pq, append=False)["n_rows_written"] == n_a
    assert not list(pq.parent.glob(".rows.parquet.tmp-*")), "the temp file of the atomic publish was left behind"
    # --paper-rows-replace with TWO waves replaces the file ONCE by both (not wave by wave, which kept only the last)
    recs = An.write_paper_rows_all([a, b], pq, replace=True)
    assert [r.n_rows_written for r in recs] == [n_a, n_a + n_b] and len(pd.read_parquet(pq)) == n_a + n_b
    An.write_analysis(a, world["it"]); An.write_analysis(b, world["main"])
    rc = An.main(["--roster", str(world["roster"]), "--analysis", str(world["it"] / "analysis.json"), "--analysis", str(world["main"] / "analysis.json"),
                  "--paper-rows", str(tmp_path / "replace.parquet"), "--paper-rows-replace"])
    assert rc == 0 and len(pd.read_parquet(tmp_path / "replace.parquet")) == n_a + n_b


def test_two_concurrent_paper_exports_keep_both_waves(world, tmp_path):
    """Two processes export two waves into ONE parquet at once. The test holds the export lock while
    both start (neither may publish while held — proof they take it), then releases: the second reads the first's
    rows and both waves survive; the file is published atomically (no temp file left, never a torn parquet)."""
    import pandas as pd
    a = _analyze(world)
    b = _analyze(world, wave="gemma4e4b_main", rung1=world["main"])
    An.write_analysis(a, world["it"]); An.write_analysis(b, world["main"])
    pq = tmp_path / "shared.parquet"
    lock = pq.with_name(pq.name + ".lock")
    argv = lambda src: [sys.executable, str(SCRIPT), "--roster", str(world["roster"]), "--analysis", str(src / "analysis.json"), "--paper-rows", str(pq)]
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "MPLBACKEND": "Agg"}
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        procs = [subprocess.Popen(argv(src), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for src in (world["it"], world["main"])]
        time.sleep(12.0)                                     # imports (pandas, matplotlib), then both block on the lock
        assert not pq.exists(), "an export published while the lock was held — the writer did not take the lock"
        assert all(p.poll() is None for p in procs), [p.communicate() for p in procs if p.poll() is not None]
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    results = [(p.wait(timeout=120), p.communicate()) for p in procs]
    assert [rc for rc, _ in results] == [0, 0], results
    df = pd.read_parquet(pq)
    n_a, n_b = len(An.paper_rows(a)), len(An.paper_rows(b))
    assert len(df) == n_a + n_b and set(df["view"]) == {"gemma4e4b_it", "gemma4e4b"}
    assert not list(pq.parent.glob(".shared.parquet.tmp-*"))
    kept = sum("kept" in out and " 0 replaced" in out for _, (out, _) in results)
    assert kept == 2 and sum(f"{n_a} kept" in out or f"{n_b} kept" in out for _, (out, _) in results) == 1


def test_pull_validates_the_roster_before_fetching_anything(world, monkeypatch, tmp_path):
    def never(*a, **k):
        raise AssertionError("modal volume get was invoked before the roster passed its own rules")
    monkeypatch.setattr(An, "_modal_get", never)
    monkeypatch.setattr(An, "modal_volume_ls", never)
    bad = roster_doc(); bad["lane"] = "another-lane"
    write_roster(world["roster"], bad)
    with pytest.raises(An.AnalysisError, match="fails widen_roster.validate.*lane 'another-lane'"):
        An.main(["--roster", str(world["roster"]), "--wave", "gemma4e4b_it", "--pull", SUFFIX_IT, "--modal-python", sys.executable,
                 "--pull-root", str(tmp_path / "pull")])


def test_summary_figures_and_results_doc(world, tmp_path):
    import matplotlib.pyplot as plt
    a = _analyze(world)
    b = _analyze(world, wave="gemma4e4b_main", rung1=world["main"])
    for x, d in ((a, world["it"]), (b, world["main"])):
        x["figures"] = {"files": An.make_figures(x, tmp_path / "figs" / x["run_name"]), "url_base": "https://example/figs/" + x["run_name"]}
        assert set(x["figures"]["files"]) == set(An.FIGURE_STEMS) and plt.get_fignums() == []
        An.write_analysis(x, d)
    s = An.summarize([world["it"] / "analysis.json", world["main"] / "analysis.json"], roster_path=world["roster"], ledger=world["ledger"])
    # the unanalysed set is every OTHER wave the shipped roster declares (state-independent: the roster's wave set moves as the
    # orchestrator onboards families — Slice D split the queued llama8b wave into llama8b_it / llama8b_main)
    from sisteer.capeval.tinycap import widen_roster as W
    every_wave = set(W.waves(W.load_roster(world["roster"])))
    assert s["analysed_waves"] == ["gemma4e4b_it", "gemma4e4b_main"]
    assert set(s["unanalysed_waves"]) == every_wave - {"gemma4e4b_it", "gemma4e4b_main"} and len(every_wave) >= 4
    # one reading per (wave, steered cell, leg); provisional = the batch-8 objects' readings — both derived from the analyses
    want_readings = sum(len(x["steered"]) * len(x["tasks"]) for x in (a, b))
    want_prov = sum(1 for x in (a, b) for c in x["steered"] for _ in x["tasks"] if x["generations"][c].get("provisional"))
    assert len(s["readings"]) == want_readings and sum(r["provisional"] for r in s["readings"]) == want_prov and want_prov == 4
    # the summary reads each leg's FINAL reading: the it wave's BCB fired and is pending → no delta, named as pending
    pending = [r for r in s["readings"] if not r["final"]]
    assert len(pending) == len(a["steered"]) and all(r["task"] == BCB and r["wave"] == "gemma4e4b_it" and r["delta_pp"] is None and r["source"] == "pending"
                                     and r["reading"].startswith("pending") for r in pending)
    assert all(r["source"] == "rung1" and r["delta_pp"] is not None for r in s["readings"] if r["final"])
    assert {w for w, _ in s["unanalysed_entries"]} == set(s["unanalysed_waves"]) and s["cost"]["gpu"]["by_wave"]["gemma4e4b_it"] == pytest.approx(1.68)
    sc = An.summary_scalars(s)
    assert sc["gemma4e4b_it/bcb200/const_lora/delta_pp"] is None and sc["gemma4e4b_it/bfcl200/const_lora/final_source"] == "rung1"
    # the it wave has FOUR steered cells pending on BCB (the constitution SI among them): the summary figure's subtitle must stay
    # within A.SUBTITLE_MAX_LINES — it names pending legs per (wave, leg) with a cell count, never cell by cell (a per-cell list overflowed)
    assert len(pending) >= 4
    files = An.make_figures(s, tmp_path / "figs" / "summary", figures=An.SUMMARY_FIGURES)
    assert set(files) == set(An.SUMMARY_FIGURE_STEMS) and plt.get_fignums() == []
    with pytest.raises(An.AnalysisError, match="two analyses for wave"):
        An.summarize([world["it"] / "analysis.json", world["it"] / "analysis.json"], roster_path=world["roster"], ledger=world["ledger"])
    md = An.render_results_doc([a, b], roster=world["doc"], summary=s, wandb_urls={"gemma4e4b_it": "https://wandb.example/r"})
    for needle in ("## Wave `gemma4e4b_it`", "## Wave `gemma4e4b_main`", "## Wave `qwen35_4b_main`", An.PENDING, "**provisional**", "gemma4e4b_it_doc_eb64",
                   "## Cross-wave summary", "Reading:", "no cross-object contrast", "clamp override", "T=0.001", "PARTIAL", "W&B: https://wandb.example/r",
                   "Roster and generations", "Decoding protocol", "WEAK gemma residual", "25.0 h ceiling", "Final-reading rule", "48 columns",
                   "object_artifact_sha256", "Headline (final readings)", "**no final reading**", "(final: rung1)", "| pending |", "RE-DERIVED from the roster"):
        assert needle in md, needle
    sk = An.render_results_doc([], roster=world["doc"], skeleton=True)
    assert sk.count(An.PENDING) >= 4 and "alphasteer_as_fit" in sk and "skeleton" in sk
    sk2 = An.render_results_doc([], roster=None, skeleton=True)
    assert all(name in sk2 for name, _ in An.PLANNED_WAVES) and "PSR is out by default" in sk2
    # the SHIPPED roster's skeleton names the two PSR rows as out of the paper, pins intact; the constitution SI runs
    shipped = An.render_results_doc([], roster=roster_doc(), skeleton=True)
    for cell in (CONST_PSR, DIFF_PSR):
        assert f"| {cell} | {An.PSR_OUT_LABEL} | {An.PSR_OUT_RULING_PREFIX}" in shipped
    # five PSR entries read as out: the two pinned ones + the three gemma-it ones moved from `queued` to blocked under the rule
    assert "PSR is out of the paper by default" in shipped and shipped.count(An.PSR_OUT_LABEL) == 5 and f"| const_si | {CONST_SI} | runnable |" in shipped


def test_cli_end_to_end_with_wandb_disabled(world, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(An.FIGS_URL_BASE_ENV, raising=False)         # no public mirror URL: the unset-variable case below
    out_md, pq = tmp_path / "results.md", tmp_path / "rows.parquet"
    rc = An.main(["--roster", str(world["roster"]), "--wave", "gemma4e4b_it", "--analyze", str(world["it"]), "--repo-root", str(world["repo"]),
                  "--ledger", str(world["ledger"]), "--no-wandb", "--figs-root", str(tmp_path / "figs"), "--write-results", str(out_md), "--paper-rows", str(pq)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gemma4e4b_it BCB-200: NO final reading" in out and "gemma4e4b_it BFCL-200 const_lora (prov.) (final: rung1)" in out
    analysis = json.loads((world["it"] / An.ANALYSIS_FILENAME).read_text())
    assert "9 figures" in out and "NOT MET (ladder pending)" in out and f"{expected_paper_rows(analysis)} rows added" in out
    assert analysis["schema"] == 2 and analysis["wandb"] is None and analysis["figures"]["url_base"].endswith("/" + SUFFIX_IT)
    assert analysis["final"][BCB]["final"] is False and analysis["final"][BFCL]["source"] == "rung1"
    assert out_md.is_file() and "## Wave `gemma4e4b_it`" in out_md.read_text() and pq.is_file()
    assert (tmp_path / "figs" / SUFFIX_IT / "final_paired_delta.svg").is_file()
    # with no public mirror URL configured, every figure link is the figure's absolute path: analysis.json's url_base and the
    # results doc (written anywhere) both resolve to the file that was drawn
    assert analysis["figures"]["url_base"] == str((tmp_path / "figs").resolve() / SUFFIX_IT)
    links = [ln.split(" — ", 1)[1].strip() for ln in out_md.read_text().splitlines() if ln.startswith("- `final_") and " — " in ln]
    linked = [link for link in links if not link.startswith("_pending")]       # the unlanded waves' placeholder lines carry no link
    assert len(linked) == len(An.FIGURE_STEMS) and all(Path(link).is_absolute() and Path(link).is_file() for link in linked), linked[:3]
    assert An.figs_url_base_for(tmp_path / "figs") == str((tmp_path / "figs").resolve())
    monkeypatch.setenv(An.FIGS_URL_BASE_ENV, "https://example/figs/")
    assert An.figs_url_base_for(tmp_path / "figs") == f"https://example/figs/{An.LANE}"
    monkeypatch.delenv(An.FIGS_URL_BASE_ENV)
    rc = An.main(["--roster", str(world["roster"]), "--summary", str(world["it"] / "analysis.json"), "--ledger", str(world["ledger"]), "--no-wandb",
                  "--figs-root", str(tmp_path / "figs"), "--write-results", str(tmp_path / "all.md")])
    assert rc == 0 and (tmp_path / "figs" / "summary" / "summary.json").is_file() and "## Cross-wave summary" in (tmp_path / "all.md").read_text()
    rc = An.main(["--roster", str(tmp_path / "absent.yaml"), "--write-results", str(tmp_path / "sk.md"), "--skeleton"])
    assert rc == 0 and An.PENDING in (tmp_path / "sk.md").read_text()
    with pytest.raises(An.AnalysisError, match="need --wave"):
        An.main(["--roster", str(world["roster"]), "--analyze", str(world["it"]), "--no-wandb", "--no-figs"])


def test_remote_paths_follow_the_store_layout_per_row():
    assert An.remote_cell_path("bcbw-gemma4e4b_it-abc1234", NONE, model_slug="gemma4e4b-f22it") == "tinycap/gen/gemma4e4b-f22it/none/thinking_off/bcbw-gemma4e4b_it-abc1234"
    assert An.remote_ladder_dir("s", model_slug="nemotron3nano4b") == "tinycap/gen/nemotron3nano4b/_ladder/s"
    # the object dir on the volume is the BARE cell id (round 6) — never the roster's staged_id (`derived__…`)
    assert An.remote_sidecar_path("/vol/objects/sdfbase/full22/qwen35_4b/main", DIFF_LORA, "derived_meta.json") == f"objects/sdfbase/full22/qwen35_4b/main/{DIFF_LORA}/derived_meta.json"
    for wname, model, view in (("nemotron3nano4b_main", "nemotron3nano4b", "main"), ("llama8b_it", "llama8b", "it"), ("llama8b_main", "llama8b", "main"), ("qwen35_4b_main", "qwen35_4b", "main")):
        spec = An.wave_spec(roster_doc(), wname)
        e = spec.entry_of_cell(DIFF_LORA)
        assert An.remote_sidecar_path(spec.objects_dir, e.cell_id, An.sidecar_filename(e)) == f"objects/sdfbase/full22/{model}/{view}/{DIFF_LORA}/derived_meta.json"
        assert An.remote_sidecar_path(spec.objects_dir, e.cell_id, An.STAGED_FROM_FILENAME).split("/")[-2] == DIFF_LORA


def test_find_certificates_matches_the_exact_names_and_lists_superseded_ones(tmp_path):
    lad = tmp_path / "rung" / An.LADDER_SUBDIR
    lad.mkdir(parents=True)
    keep = [lad / RP.certificate_filename(CONST_LORA), lad / RP.certificate_filename(CONST_PSR), lad / RP.CERTIFICATE_FILENAME]
    gone = [RP.superseded_certificate_path(lad / RP.certificate_filename(CONST_LORA), at=T0), RP.superseded_certificate_path(lad / RP.CERTIFICATE_FILENAME, at=T0)]
    for p in keep + gone + [lad / RP.ADMISSION_FILENAME, lad / "rung_probe.notes.txt"]:
        p.write_text("{}")
    certs, superseded = An.find_certificates([tmp_path / "rung"], explicit=[tmp_path / "explicit.json"])
    assert certs == [tmp_path / "explicit.json", *sorted(keep)] and superseded == sorted(gone)
    assert all(".superseded." not in p.name for p in certs)


def test_suffix_grammar_and_run_id_grammar():
    assert An.parse_suffix(f"bcbw-gemma4e4b_it-{SHA7}") == {"wave": "gemma4e4b_it", "sha7": SHA7, "tag": None}
    assert An.parse_suffix(f"bcbw-qwen35_4b_main-{SHA7}-cap16k") == {"wave": "qwen35_4b_main", "sha7": SHA7, "tag": "cap16k"}
    for bad in ("bcbbfcl200-5039ce9", "bcbw-gemma4e4b_it-5039ce", "gemma4e4b_it-5039ce9", ""):
        with pytest.raises(An.AnalysisError, match="is not the lane's"):
            An.parse_suffix(bad)
    assert An.RUN_ID_RE.match(f"{SHA7}-deadbeef") and not An.RUN_ID_RE.match("5039ce9-dead") and not An.RUN_ID_RE.match("11111111-2222-3333-4444-555555555555")


# --------------------------------------------------------------------------- #
# publication eligibility: a saved analysis under the CURRENT roster (round 5)  #
# --------------------------------------------------------------------------- #


def test_a_saved_analysis_is_gated_by_the_current_roster_before_publication(world, tmp_path):
    """`summarize` and `--analysis` loaded saved analysis.json records and published every
    paired cell they carried — an earlier analysis with PSR readings republished PSR as a final measured result into the summary,
    the figures, the results doc and the paper export. Every saved analysis is now gated by the CURRENT roster's publication
    eligibility (`gate_saved_analysis`): a measured cell whose entry is blocked (or gone) is suppressed in the summary, the figures
    and the doc — named `out (…)` with the status and reason, never silently dropped — and the paper export REFUSES naming it; the
    saved file is never rewritten; the same analysis under a roster that runs the cell publishes it."""
    import hashlib
    import pandas as pd
    a = _analyze(world)                                            # the fixture roster RUNS PSR: measured, stamped eligible
    assert a["publication"]["out"] == {} and a["publication"]["cells_at_analysis"] == a["cells"] and CONST_PSR in a["cells"]
    assert a["publication"]["roster_sha256"] == W.roster_sha256(world["roster"]) and a["publication"]["unmeasured_runnable"] == []
    saved = An.write_analysis(a, world["it"])
    digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    # the CURRENT roster now blocks PSR (the shipped state) while the saved record still carries the PSR reading
    write_roster(world["roster"], roster_doc())
    shipped = An.load_roster(world["roster"])
    out = An.publication_eligibility(a, shipped)
    assert list(out) == [CONST_PSR] and out[CONST_PSR]["status"] == "blocked" and out[CONST_PSR]["label"] == An.PSR_OUT_LABEL
    assert out[CONST_PSR]["reason"].startswith(An.PSR_OUT_RULING_PREFIX) and out[CONST_PSR]["entry_id"] == "const_psr"
    s = An.summarize([saved], roster_path=world["roster"], ledger=world["ledger"])
    (v,) = s["waves"]
    assert CONST_PSR not in v["cells"] and CONST_PSR not in v["steered"] and CONST_PSR not in v["fitted"] and v["cells"] == [c for c in a["cells"] if c != CONST_PSR]
    assert v["publication"]["out"] == out and v["publication"]["roster_sha256"] == W.roster_sha256(world["roster"]) and v["publication"]["cells_at_analysis"] == a["cells"]
    for task in v["legs"]:
        assert CONST_PSR not in v["legs"][task]["cells"] and CONST_PSR not in v["legs"][task]["paired"]
        assert v["final"][task]["cells"] is None or CONST_PSR not in v["final"][task]["cells"]
    assert all(CONST_PSR not in r["deltas"] and CONST_PSR not in r["cells"] for bd in v["breakdowns"].values() for r in bd["rows"])
    assert all(CONST_PSR not in q["cells"] for q in v["quoted"].values()) and CONST_PSR in v["generations"] and CONST_PSR in v["identity"]["provenance"]["attempts_by_cell"]
    psr = [r for r in s["readings"] if r["cell"] == CONST_PSR]
    assert len(psr) == len(v["tasks"]) and all(r["out"] and r["source"] == An.OUT_SOURCE and r["delta_pp"] is None and not r["final"] for r in psr)
    assert all(r["reading"].startswith(An.PSR_OUT_LABEL) and r["out_status"] == "blocked" for r in psr)
    assert all(not r["out"] for r in s["readings"] if r["cell"] != CONST_PSR) and s["out"] == {"gemma4e4b_it": out}
    sc = An.summary_scalars(s)
    assert sc["gemma4e4b_it/bfcl200/const_psr/delta_pp"] is None and sc["gemma4e4b_it/bfcl200/const_psr/final_source"] == "out"
    assert sc["gemma4e4b_it/bfcl200/const_lora/delta_pp"] is not None
    # the view's roster record carries the CURRENT status (the status at analysis kept beside it); figures draw without the cell
    e = next(x for x in v["wave"]["entries"] if x["cell_id"] == CONST_PSR)
    assert e["status"] == "blocked" and e["status_at_analysis"] == "runnable" and e["reason"].startswith(An.PSR_OUT_RULING_PREFIX)
    assert set(An.make_figures(s, tmp_path / "figs" / "summary", figures=An.SUMMARY_FIGURES)) == set(An.SUMMARY_FIGURE_STEMS)
    assert set(An.make_figures(v, tmp_path / "figs" / "wave")) == set(An.FIGURE_STEMS)
    # the results doc names the ruling in the wave section, the headline and the cross-wave table; no PSR contrast anywhere
    md = An.render_results_doc(s["waves"], roster=shipped, summary=s)
    assert f"`{CONST_PSR}` — **{An.PSR_OUT_LABEL}**" in md and "Not published under the current roster" in md
    headline = md.split("# Headline (final readings)")[1].split("# Decoding protocol")[0]
    assert f"(`{CONST_PSR}`): **{An.PSR_OUT_LABEL}**" in headline and all("paired difference" not in ln for ln in headline.splitlines() if CONST_PSR in ln)
    cross = md.split("## Cross-wave summary")[1].split("## Wave")[0]
    assert any("const_psr" in ln and "| out |" in ln for ln in cross.splitlines()) and "paired difference" not in cross
    # the paper export REFUSES the view — direct, via --analysis and via --summary — and nothing of PSR reaches the parquet
    pq = tmp_path / "rows.parquet"
    with pytest.raises(An.AnalysisError, match=f"cell '{CONST_PSR}' is blocked in the current roster"):
        An.write_paper_rows(v, pq)
    assert not pq.exists()
    with pytest.raises(An.AnalysisError, match="blocked in the current roster"):
        An.main(["--roster", str(world["roster"]), "--analysis", str(saved), "--paper-rows", str(pq)])
    with pytest.raises(An.AnalysisError, match="blocked in the current roster"):
        An.main(["--roster", str(world["roster"]), "--summary", str(saved), "--ledger", str(world["ledger"]), "--no-wandb",
                 "--figs-root", str(tmp_path / "figs2"), "--paper-rows", str(pq)])
    assert not pq.exists()
    # --analysis + --write-results publishes the doc WITH the out row (no export asked); the saved record is byte-identical throughout
    rc = An.main(["--roster", str(world["roster"]), "--analysis", str(saved), "--write-results", str(tmp_path / "r.md")])
    assert rc == 0 and An.PSR_OUT_LABEL in (tmp_path / "r.md").read_text()
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
    # the record's OWN stamp (made when PSR was runnable) is stale: re-gating redoes it from cells_at_analysis; an ungated record refuses export
    raw = json.loads(saved.read_text())
    assert raw["publication"]["out"] == {}
    regated = An.gate_saved_analysis(raw, shipped, roster_path=world["roster"])
    assert regated["publication"]["out"] == out and CONST_PSR not in regated["cells"] and raw["cells"] == a["cells"]
    assert An.gate_saved_analysis(regated, shipped)["cells"] == regated["cells"]                    # idempotent on the view
    raw.pop("publication")
    with pytest.raises(An.AnalysisError, match="not gated against the current roster"):
        An.paper_rows(raw)
    # a cell whose entry is GONE is out too; a roster without the wave, or without a runnable `none`, refuses outright
    gone = roster_doc(); gone["waves"]["gemma4e4b_it"]["entries"].pop("const_psr")
    g = An.publication_eligibility(a, gone)
    assert g[CONST_PSR]["status"] == An.ABSENT_STATUS and g[CONST_PSR]["label"] == An.ABSENT_LABEL and g[CONST_PSR]["entry_id"] is None
    nowave = roster_doc(); nowave["waves"].pop("gemma4e4b_it")
    with pytest.raises(An.AnalysisError, match="has no wave 'gemma4e4b_it'"):
        An.publication_eligibility(a, nowave)
    noanchor = roster_doc(); noanchor["waves"]["gemma4e4b_it"]["entries"]["none"].update(status="blocked", reason="x", unblock="y")
    with pytest.raises(An.AnalysisError, match="does not run its `none` anchor"):
        An.publication_eligibility(a, noanchor)
    # the SAME saved analysis under a roster that RUNS PSR publishes it: readings with a delta, paper rows with the PSR cell
    write_roster(world["roster"], fixture_roster())
    s2 = An.summarize([saved], roster_path=world["roster"], ledger=world["ledger"])
    assert s2["waves"][0]["publication"]["out"] == {} and CONST_PSR in s2["waves"][0]["cells"] and s2["out"] == {"gemma4e4b_it": {}}
    psr2 = [r for r in s2["readings"] if r["cell"] == CONST_PSR]
    assert len(psr2) == len(a["tasks"]) and any(r["delta_pp"] is not None for r in psr2) and not any(r["out"] for r in psr2)
    rec = An.write_paper_rows(s2["waves"][0], pq)
    df = pd.read_parquet(pq)
    assert rec["n_added"] == len(df) and CONST_PSR in set(df["cell_id"])


# --------------------------------------------------------------------------- #
# --pull: the object sidecar from the volume's CELL-ID dir (round 6)            #
# --------------------------------------------------------------------------- #

SUFFIX_NEM, RUN_NEM = f"bcbw-nemotron3nano4b_main-{SHA7}", f"{SHA7}-44444444"
NEM_OBJECTS_DIR = "objects/sdfbase/full22/nemotron3nano4b/main"


def fake_volume(vol: Path, world) -> Tuple[Path, Dict[str, Any]]:
    """The volume as it IS (listed 2026-09-12): every runnable cell's store root under `tinycap/gen/<row>/<cell>/thinking_off/<suffix>/`
    and the staged object under `objects/sdfbase/full22/<model>/<view>/<CELL ID>/{object.pt, derived_meta.json, staged_from.json}` —
    a bare cell-id dir; the roster's `derived__…` staged_id exists on the volume ONLY inside staged_from.json."""
    doc = world["doc"]
    spec = An.wave_spec(doc, "nemotron3nano4b_main")
    build = make_wave(vol.parent / "build" / SUFFIX_NEM, doc, "nemotron3nano4b_main", roster_path=world["roster"], run_id=RUN_NEM)
    for cell in spec.cells:
        dst = vol / An.remote_cell_path(SUFFIX_NEM, cell, model_slug=str(spec.model_row))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(build / cell, dst)
    e = spec.entry_of_cell(DIFF_LORA)
    sha = e.object["sha256"]
    obj_dir = vol / NEM_OBJECTS_DIR / DIFF_LORA
    obj_dir.mkdir(parents=True)
    (obj_dir / "object.pt").write_bytes(b"not a tensor; the sha the roster pins is the real object's")
    meta = json.dumps({"layers": An._declared_layers(e.object["fit"]), "object_sha256": sha, "kind": "lora_subtraction", "model_id": spec.served["hf"],
                       "positions": "all", "parent_pos": {"object_sha256": "a" * 64}, "parent_neg": {"object_sha256": "b" * 64}}, indent=2).encode()
    (obj_dir / "derived_meta.json").write_bytes(meta)
    record = {"cell_id": DIFF_LORA, "lane": "derived", "object_sha256": sha, "pulled_at": "2026-09-08T05:54:50Z",
              "recorded_hash_check": {"entries": [{"file": "derived_meta.json", "key": "object_sha256", "matches_object": True, "value": sha}], "status": "match"},
              "source": "cluster_a:$SHARED/exp/objects/sdfbase/newmodels_full/derived/lora_subtraction/nemotron3nano4b_doc/" + DIFF_LORA,
              "staged_by": "stage_full22.sh", "staged_id": e.object["staged_id"], "target": "nemotron3nano4b",
              "staged_shas": {"derived_meta.json": hashlib.sha256(meta).hexdigest(), "object.pt": sha}}
    (obj_dir / An.STAGED_FROM_FILENAME).write_text(json.dumps(record, indent=2))
    return build, record


def fake_modal(vol: Path, requests: List[str]):
    """`modal volume get --force <vol> <remote> <dest>` lands a dir as `<dest>/<basename>` (recursive) and a file as `<dest>/<basename>`;
    a missing path exits 1 with the CLI's error. `modal volume ls` lists one level of names, refusing a missing path."""
    from types import SimpleNamespace

    def get(modal_python, volume, remote, dest):
        requests.append(remote)
        src = vol / remote
        if src.is_dir():
            shutil.copytree(src, Path(dest) / src.name, dirs_exist_ok=True)
        elif src.is_file():
            Path(dest).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, Path(dest) / src.name)
        else:
            return SimpleNamespace(returncode=1, stdout="", stderr=f"╭─ Error ─╮\n│ Path {remote} not found in volume {volume} │\n╰─╯")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def ls(modal_python, volume, remote):
        src = vol / remote
        if not src.is_dir():
            raise An.AnalysisError(f"modal volume ls failed (rc 1): path {remote} not found")
        return sorted(p.name for p in src.iterdir())
    return get, ls


def test_pull_fetches_the_object_sidecar_from_the_volumes_cell_id_dir_and_checks_the_staging_record(world, tmp_path, monkeypatch):
    """Round 6 (live): `--pull` refused nemotron3nano4b_main and llama8b_it at 039394c4 as INCOMPLETE — the sidecar path was composed
    from the roster's `staged_id` (`objects/…/main/derived__constv2-mix-full__lora_r4_ce_minus_ctl/derived_meta.json`) while the volume
    keys every staged object by its BARE cell id and records the staged_id INSIDE the dir's staged_from.json. The pull now fetches
    `<objects_dir>/<cell id>/{<sidecar>, staged_from.json}` and refuses unless the record's staged_id, cell_id and object sha256 are the
    roster's and the landed sidecar's bytes are the ones the record lists; the pulled dir analyses end to end."""
    vol = tmp_path / "volume"
    build, record = fake_volume(vol, world)
    assert not (vol / NEM_OBJECTS_DIR / f"derived__{DIFF_LORA}").exists() and (vol / NEM_OBJECTS_DIR / DIFF_LORA / An.STAGED_FROM_FILENAME).is_file()
    requests: List[str] = []
    get, ls = fake_modal(vol, requests)
    monkeypatch.setattr(An, "_modal_get", get)
    monkeypatch.setattr(An, "modal_volume_ls", ls)
    spec = An.wave_spec(world["doc"], "nemotron3nano4b_main")
    lines: List[str] = []
    dest = An.pull_suffix(SUFFIX_NEM, spec, dest_root=tmp_path / "pulled", modal_python=sys.executable, log=lines.append)
    assert dest == tmp_path / "pulled" / SUFFIX_NEM
    # the store roots landed and verified; the sidecar AND the staging record came from the cell-id dir; no request ever named the staged_id
    assert (dest / NONE / "cells").is_dir() and (dest / DIFF_LORA / "cells").is_dir() and (dest / An.LADDER_SUBDIR).is_dir()
    side = dest / An.OBJECTS_SUBDIR / DIFF_LORA
    assert (side / "derived_meta.json").read_bytes() == (vol / NEM_OBJECTS_DIR / DIFF_LORA / "derived_meta.json").read_bytes()
    assert json.loads((side / An.STAGED_FROM_FILENAME).read_text()) == record
    assert f"{NEM_OBJECTS_DIR}/{DIFF_LORA}/derived_meta.json" in requests and f"{NEM_OBJECTS_DIR}/{DIFF_LORA}/{An.STAGED_FROM_FILENAME}" in requests
    assert not any("derived__" in r for r in requests), requests
    assert any(f"staged_id {record['staged_id']} = the volume's" in ln for ln in lines) and any("no ladder dir" in ln for ln in lines)
    # the pulled dir analyses end to end; the fit layers come from the pulled sidecar and the staged_id is recorded from the volume's record
    a = An.analyze(dest, roster_path=world["roster"], wave_name="nemotron3nano4b_main", repo_root=world["repo"], ledger=world["ledger"],
                   sandbox_state_root=tmp_path / "sandbox-state")
    fl = a["fit_layers"][DIFF_LORA]
    assert fl["fit_layers"] == "12,17,24,32" and fl["staged_id"] == fl["staged_id_volume"] == record["staged_id"] and fl["staged_from"] == str(side / An.STAGED_FROM_FILENAME)
    assert a["cells"] == [NONE, DIFF_LORA] and a["tasks"] == [BCB]
    # the volume dir holding ANOTHER staging (staged_id / cell / object off the roster's) refuses by name; so does a missing record and a sidecar
    # whose bytes are not the ones the record lists; the pure check names every disagreement
    e = spec.entry_of_cell(DIFF_LORA)
    meta_sha = hashlib.sha256((vol / NEM_OBJECTS_DIR / DIFF_LORA / "derived_meta.json").read_bytes()).hexdigest()
    assert An.staged_from_problems(record, e, sidecar="derived_meta.json", sidecar_sha256=meta_sha) == []
    bad = {**record, "staged_id": "derived_itdoc__" + DIFF_LORA, "cell_id": "constv2-mix-full__lora_r4_ce", "object_sha256": "f" * 64}
    found = An.staged_from_problems(bad, e, sidecar="derived_meta.json", sidecar_sha256="0" * 64)
    assert len(found) == 4 and "staged_id 'derived_itdoc__" in found[0] and "!= the roster's 'derived__" in found[0] and "cell_id" in found[1]
    assert "object_sha256" in found[2] and "staged_shas[derived_meta.json]" in found[3]
    assert An.staged_from_problems({"cell_id": DIFF_LORA}, e, sidecar="derived_meta.json", sidecar_sha256=meta_sha) == [f"{DIFF_LORA}: {An.STAGED_FROM_FILENAME} names no staged_id (keys ['cell_id'])"]
    assert An.staged_from_problems("nope", e, sidecar="derived_meta.json", sidecar_sha256=meta_sha) == [f"{DIFF_LORA}: {An.STAGED_FROM_FILENAME} is not a JSON object"]
    (vol / NEM_OBJECTS_DIR / DIFF_LORA / An.STAGED_FROM_FILENAME).write_text(json.dumps(bad))
    with pytest.raises(An.AnalysisError, match=f"INCOMPLETE .*{DIFF_LORA}: {An.STAGED_FROM_FILENAME} staged_id 'derived_itdoc__.* != the roster's 'derived__"):
        An.pull_suffix(SUFFIX_NEM, spec, dest_root=tmp_path / "pulled2", modal_python=sys.executable, log=lines.append)
    (vol / NEM_OBJECTS_DIR / DIFF_LORA / An.STAGED_FROM_FILENAME).unlink()
    with pytest.raises(An.AnalysisError, match=f"INCOMPLETE .*{DIFF_LORA}: staging record {NEM_OBJECTS_DIR}/{DIFF_LORA}/{An.STAGED_FROM_FILENAME} did not land"):
        An.pull_suffix(SUFFIX_NEM, spec, dest_root=tmp_path / "pulled3", modal_python=sys.executable, log=lines.append)
    (vol / NEM_OBJECTS_DIR / DIFF_LORA / An.STAGED_FROM_FILENAME).write_text(json.dumps(record))
    (vol / NEM_OBJECTS_DIR / DIFF_LORA / "derived_meta.json").write_text(json.dumps({"layers": [12, 17, 24, 32], "object_sha256": e.object["sha256"], "edited": True}))
    with pytest.raises(An.AnalysisError, match=r"INCOMPLETE .*staged_shas\[derived_meta.json\] .* != the sha256 of the derived_meta.json that landed"):
        An.pull_suffix(SUFFIX_NEM, spec, dest_root=tmp_path / "pulled4", modal_python=sys.executable, log=lines.append)
    # the analysis applies the same check to a pulled dir whose record disagrees with the sidecar beside it
    (side / An.STAGED_FROM_FILENAME).write_text(json.dumps({**record, "staged_id": "derived_itdoc__" + DIFF_LORA}))
    with pytest.raises(An.AnalysisError, match="staged_id 'derived_itdoc__.* != the roster's"):
        An.fit_layers_of(dest, e, served_sha=e.object["sha256"])
    (side / An.STAGED_FROM_FILENAME).unlink()                      # an older pull without the record: the sidecar alone still reads (recorded as unverified)
    assert An.fit_layers_of(dest, e, served_sha=e.object["sha256"])["staged_id_volume"] is None


# --------------------------------------------------------------------------- #
# a gated view's FIGURES: withheld or regenerated, never the pre-ruling ones    #
# --------------------------------------------------------------------------- #


def _figure_texts(fig) -> List[str]:
    texts = [t.get_text() for t in fig.texts] + [t.get_text() for ax in fig.axes for t in ax.texts]
    if getattr(fig, "_suptitle", None) is not None:
        texts.append(fig._suptitle.get_text())
    return texts


def test_a_gated_views_figures_are_withheld_or_regenerated_never_the_pre_ruling_ones(world, tmp_path):
    """Round 7: the gate removed the measurements but kept `figures`, and the renderer (with its `<url base>/<suffix>/`
    fallback) linked the analysis-time `final_paired_delta.png` etc. — plots that still contain the out cell — under a doc that says the
    cell is suppressed. Now the gate WITHHOLDS the figures (no URL of any kind, the reason stated) and `--analysis` / `--summary` with
    figures enabled REGENERATE them from the gated view under `<figs-root>/<suffix>/gated-<roster sha8>/`, linking those; the
    analysis-time assets and the saved analysis.json stay untouched as provenance; the regenerated figures carry the out label."""
    import hashlib as _h
    import matplotlib.pyplot as plt
    a = _analyze(world)
    orig_dir = tmp_path / "figs" / a["run_name"]
    a["figures"] = {"dir": str(orig_dir), "files": An.make_figures(a, orig_dir), "url_base": "https://example/figs/" + a["run_name"]}
    saved = An.write_analysis(a, world["it"])
    orig_png = orig_dir / "final_paired_delta.png"
    before = (_h.sha256(orig_png.read_bytes()).hexdigest(), _h.sha256(saved.read_bytes()).hexdigest())
    write_roster(world["roster"], roster_doc())                    # PSR out under the current roster
    shipped = An.load_roster(world["roster"])
    v = An.gate_saved_analysis(json.loads(saved.read_text()), shipped, roster_path=world["roster"])
    sha8 = W.roster_sha256(world["roster"])[:8]
    assert v["figures"]["withheld"] is True and v["figures"]["out"] == [CONST_PSR] and "url_base" not in v["figures"]
    assert v["publication"]["figures_at_analysis"]["url_base"] == "https://example/figs/" + a["run_name"]
    # rendered without regeneration: NO figure URL for the wave — not the analysis-time ones, not the renderer's fallback
    md = An.render_results_doc([v], roster=shipped)
    wave_md = md.split("## Wave `gemma4e4b_it`")[1].split("## Wave `")[0]
    assert "Figures withheld" in wave_md and "final_paired_delta.png" not in wave_md and ".png" not in wave_md
    assert "https://example/figs/" not in md and f"{An.DEFAULT_FIGS_URL_BASE.rstrip('/')}/{a['run_name']}" not in md
    # --analysis with figures OFF: withheld; with figures ON: regenerated from the gated view under the gated namespace and linked
    rc = An.main(["--roster", str(world["roster"]), "--analysis", str(saved), "--write-results", str(tmp_path / "off.md"), "--no-figs"])
    off = (tmp_path / "off.md").read_text()
    assert rc == 0 and "Figures withheld" in off and ".png" not in off.split("## Wave `gemma4e4b_it`")[1]
    rc = An.main(["--roster", str(world["roster"]), "--analysis", str(saved), "--write-results", str(tmp_path / "on.md"),
                  "--figs-root", str(tmp_path / "figs2"), "--figs-url-base", "https://example/figs"])
    on = (tmp_path / "on.md").read_text()
    gated_dir = tmp_path / "figs2" / a["run_name"] / f"gated-{sha8}"
    assert rc == 0 and gated_dir.is_dir() and set(p.stem for p in gated_dir.glob("final_*.png")) == {f"final_{s}" for s in An.FIGURE_STEMS}
    assert f"https://example/figs/{a['run_name']}/gated-{sha8}/final_paired_delta.png" in on and "regenerated from the GATED view" in on
    assert f"https://example/figs/{a['run_name']}/final_" not in on and "Figures withheld" not in on.split("## Wave `gemma4e4b_it`")[1]
    assert An.gated_figs_dir(v, tmp_path / "figs2") == gated_dir
    # --summary regenerates the per-wave figures of a gated wave the same way (and its own summary figures)
    rc = An.main(["--roster", str(world["roster"]), "--summary", str(saved), "--ledger", str(world["ledger"]), "--no-wandb",
                  "--figs-root", str(tmp_path / "figs3"), "--figs-url-base", "https://example/figs", "--write-results", str(tmp_path / "sum.md")])
    sm = (tmp_path / "sum.md").read_text()
    assert rc == 0 and f"https://example/figs/{a['run_name']}/gated-{sha8}/final_roster_status.png" in sm and f"https://example/figs/{a['run_name']}/final_" not in sm
    assert (tmp_path / "figs3" / a["run_name"] / f"gated-{sha8}" / "final_paired_delta.png").is_file() and (tmp_path / "figs3" / "summary" / "final_summary_paired_delta.png").is_file()
    # the regenerated figures carry the out label; the analysis-time assets and the saved record are byte-identical
    fig = An.fig_roster_status(v)
    assert any(t.startswith(An.PSR_OUT_LABEL) for t in _figure_texts(fig)); plt.close(fig)
    fig = An.fig_paired_delta(v)
    assert any(" · 1 out" in t for t in _figure_texts(fig)); plt.close(fig)
    fig = An.fig_pass_rate_by_cell(v)
    assert any(" · 1 out" in t for t in _figure_texts(fig)); plt.close(fig)
    assert (_h.sha256(orig_png.read_bytes()).hexdigest(), _h.sha256(saved.read_bytes()).hexdigest()) == before
    # a view already regenerated is left alone by the gate (idempotent) and by a second regeneration
    g = An.gate_saved_analysis(json.loads(saved.read_text()), shipped, roster_path=world["roster"])
    assert An.regenerate_gated_figures(g, figs_root=tmp_path / "figs4", figs_url_base="https://example/figs")["gated"]["out"] == [CONST_PSR]
    again = An.gate_saved_analysis(g, shipped, roster_path=world["roster"])
    assert again["figures"] == g["figures"] and An.regenerate_gated_figures(again, figs_root=tmp_path / "figs4") is None
    # a fresh analysis under a roster that runs every measured cell keeps its own figures (nothing withheld)
    write_roster(world["roster"], fixture_roster())
    fresh = An.gate_saved_analysis(json.loads(saved.read_text()), An.load_roster(world["roster"]), roster_path=world["roster"])
    assert fresh["figures"]["url_base"] == "https://example/figs/" + a["run_name"] and "withheld" not in fresh["figures"]


# --------------------------------------------------------------------------- #
# the ladder's tree admission reaches the analysis (bcb-bfcl-200-widen)       #
# --------------------------------------------------------------------------- #

TREE_FIXED = "d73d73d7" + "1" * 32
WHY_FIXED = "rung 2 re-scored at the tree carrying the BCB verifier transport fix (D73)"


def _merged_main_world(tmp_path, *, rung2_tree: str, admit: bool) -> Dict[str, Any]:
    """The gemma4e4b_main wave of `test_bcb_bfcl_200_analysis` (three cells, BCB fired) with rung 2 produced at
    ``rung2_tree`` and merged — with the tree admission when ``admit``. The sibling module is imported here, not at
    the top: it imports THIS module as ``W``."""
    import test_bcb_bfcl_200_analysis as A
    world = A.make_world(tmp_path)
    base, lora, psr = A.NONE, A.LORA, A.PSR
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel = L.write_selection(doc, tmp_path / "ladder_selection.json")
    rung2 = F.make_rung2(tmp_path / "rung2", doc, residual={(lora, BCB): {F.CAPPED_BCB[2]}}, cells=list(A.CELLS),
                         store_kwargs={"tree_sha": rung2_tree},
                         store_kwargs_by_cell={lora: {"object_sha256": SHA[lora]}, psr: {"object_sha256": SHA[psr]}, base: {"object_sha256": None}},
                         values=lambda c, t, i: wave_values(c)(c, t, i))
    spec = An.wave_spec(world["doc"], A.WAVE)
    for cell in A.CELLS:
        obj = spec.entry_of_cell(cell).object
        protocolize_store(L.store_dir_for(rung2, cell, BCB), tasks=[BCB], served_sha=(None if cell == base else SHA[cell]), cell=cell,
                          kind=(None if obj is None else obj.get("kind")), arm=(None if cell == base else cell.replace("__", "--")), null_route_digest=False)
    lad = tmp_path / "rung2" / RP.LADDER_SUBDIR
    certs, cert_docs = {}, {}
    for cell in (lora, psr):
        p = F.make_certificate(lad / RP.certificate_filename(cell), doc, object_digests={lora: SHA[lora], psr: SHA[psr]}, accepted=(32_768, 16),
                               tree_sha=rung2_tree)
        c = json.loads(p.read_text()); c["probed_cell"] = cell
        RP.write_certificate(p, c)
        certs[cell], cert_docs[cell] = p, c
    RP.write_admission(lad / RP.ADMISSION_FILENAME, RP.build_admission(suffix=cert_docs[lora]["suffix"], common=cert_docs[lora]["accepted"],
                                                                        certificates=cert_docs))
    kw = {"admit_tree": rung2_tree, "admit_tree_why": WHY_FIXED} if admit else {}
    merged = L.merge(world["rung1"], rung2, sel, certs[lora], tmp_path / "merged", repo_root=world["repo"], **kw)
    return {"A": A, "world": world, "sel": sel, "rung2": rung2, "merged": merged, "merge_dir": tmp_path / "merged"}


def test_the_merges_tree_admission_is_carried_into_the_analysis_and_the_wave_doc(tmp_path):
    w = _merged_main_world(tmp_path, rung2_tree=TREE_FIXED, admit=True)
    assert w["merged"]["tree_admission"] == {"rung1_tree_sha": TREE, "rung2_tree_sha": TREE_FIXED, "why": WHY_FIXED, "flag": "--admit-tree"}
    a = w["A"]._analyze(w["world"], ladder_merge=w["merge_dir"], selection=w["sel"], rung2_dir=w["rung2"])
    assert a["ladder"]["tree_admission"] == w["merged"]["tree_admission"] and a["ladder"]["applied"] is True
    assert json.loads(json.dumps(a["ladder"]))["tree_admission"]["rung2_tree_sha"] == TREE_FIXED       # analysis.json carries it as is
    md = An.render_wave_doc(a)
    assert f"rung 2 produced at `{TREE_FIXED[:8]}` — {WHY_FIXED} (rung 1 at `{TREE[:8]}`" in md


def test_a_merge_without_an_admission_carries_none_and_the_doc_prints_no_line(tmp_path):
    w = _merged_main_world(tmp_path, rung2_tree=TREE, admit=False)
    assert w["merged"]["tree_admission"] is None
    a = w["A"]._analyze(w["world"], ladder_merge=w["merge_dir"], selection=w["sel"], rung2_dir=w["rung2"])
    assert a["ladder"]["tree_admission"] is None and a["ladder"]["applied"] is True
    assert "rung 2 produced at" not in An.render_wave_doc(a)
    # and no merge at all: the key is present, None
    assert w["A"]._analyze(w["world"])["ladder"]["tree_admission"] is None


def test_the_analysis_refuses_a_tree_admission_that_is_not_this_merges(tmp_path):
    w = _merged_main_world(tmp_path, rung2_tree=TREE_FIXED, admit=True)
    path = w["merge_dir"] / L.MERGED_SUMMARY_FILENAME
    doc = json.loads(path.read_text())
    doc["tree_admission"]["rung2_tree_sha"] = "e" * 40                    # names a rung-2 tree the certificate was not issued at
    path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    with pytest.raises(An.AnalysisError, match="not the admission of this merge"):
        w["A"]._analyze(w["world"], ladder_merge=w["merge_dir"], selection=w["sel"], rung2_dir=w["rung2"])
    doc["tree_admission"] = {"rung2_tree_sha": TREE_FIXED}                # a partial record is not the merge's shape
    path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    with pytest.raises(An.AnalysisError, match="tree_admission carries"):
        w["A"]._analyze(w["world"], ladder_merge=w["merge_dir"], selection=w["sel"], rung2_dir=w["rung2"])


# --------------------------------------------------------------------------- #
# leg WITHDRAWAL by the current roster, the reading rule as the one emphasis,   #
# the paper `thinking` mode, the diagnostics rung label + merged at-cap, the    #
# roster figure's two encodings, p formats                                      #
# --------------------------------------------------------------------------- #


def _block_leg(doc: Mapping[str, Any], wave: str, atom: str, *, reason: str = "instrument failure (D79): the template renders no tools",
               unblock: str = "a tool-aware template verified in-lane, then the cells re-run") -> Dict[str, Any]:
    """The roster with ``wave``'s leg ``atom`` BLOCKED (the shipped llama shape) — a store-building variant."""
    import copy
    d = copy.deepcopy(dict(doc))
    d["waves"][wave]["legs"][atom] = {"status": "blocked", "reason": reason, "unblock": unblock}
    return d


def _figure_blob(fig, *, ticks: bool = False) -> str:
    """Every text of ``fig`` (titles are hard-wrapped by `wrap_title`, so newlines become spaces), the y/x tick labels too when
    asked — joined; the figure is CLOSED first, so a failing assertion never leaks it into the next test."""
    import matplotlib.pyplot as plt
    texts = list(_figure_texts(fig))
    if ticks:
        texts += [t.get_text() for ax in fig.axes for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels())]
    plt.close(fig)
    return " | ".join(t.replace("\n", " ") for t in texts)


def _paired(b: int, c: int, *, n: int = 200, lo: Optional[float] = None, hi: Optional[float] = None) -> Dict[str, Any]:
    """A synthetic paired block with the exact test recomputed from (b, c) and the interval given (or ±2 pp around Δ)."""
    d = (c - b) / n
    p = {"n_items": n, "delta": d, "ci_low": (lo if lo is not None else d - 0.02), "ci_high": (hi if hi is not None else d + 0.02), "B": 10_000,
         "seed": 0, "level": 0.95, "b_pass_to_fail": b, "c_fail_to_pass": c, "n_discordant": b + c, "mcnemar_exact_p": An.mcnemar_exact_p(b, c)}
    p["excludes_zero"] = p["ci_low"] > 0 or p["ci_high"] < 0
    p["cost_detected"] = An.cost_detected(p)
    p["reading"] = An.reading_rule(p)
    return p


def test_a_measured_leg_the_current_roster_blocks_is_withdrawn_never_read(world, tmp_path):
    """The llama BFCL instrument rendered no tool definitions — both cells 56/200 = exactly the irrelevance items,
    b = c = 0, CI [0, 0] — so the roster now BLOCKS the leg. A saved analysis that measured such a leg is gated by the CURRENT
    roster's leg status: `final[leg]` becomes source `withdrawn` / final False with the numbers kept VERBATIM under `withdrawn`
    (reason, roster sha, cells, paired); the headline, the cross-wave table, the figures, W&B and the paper VALUES exclude it; the
    paper rows keep the row with `status: withdrawn: <reason>` and null measurements; the doc prints a labelled WITHDRAWN block
    quoting the kept numbers with the reason. The saved file is never rewritten; under a roster that runs the leg it reads again."""
    import matplotlib.pyplot as plt
    import pandas as pd
    b = _analyze(world, wave="gemma4e4b_main", rung1=world["main"])            # both legs measured, neither fired: two rung-1 final readings
    assert b["final"][BFCL]["final"] and b[An.WITHDRAWN_LEGS_KEY] == {} and b["publication"][An.WITHDRAWN_LEGS_KEY] == {}
    # the record's analysis-time figures (drawn with BFCL as a reading): the gate must WITHHOLD them — a record without figures gets no marker
    b["figures"] = {"dir": str(tmp_path / "figs0" / b["run_name"]), "files": An.make_figures(b, tmp_path / "figs0" / b["run_name"]),
                    "url_base": "https://example/figs/" + b["run_name"]}
    saved = An.write_analysis(b, world["main"])
    digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    kept_cells, kept_paired = b["final"][BFCL]["cells"], b["final"][BFCL]["paired"]
    reason, unblock = "instrument failure (D79): the template renders no tools", "a tool-aware template verified in-lane, then the cells re-run"
    write_roster(world["roster"], _block_leg(fixture_roster(), "gemma4e4b_main", "bfcl_200", reason=reason, unblock=unblock))
    roster, sha = An.load_roster(world["roster"]), W.roster_sha256(world["roster"])
    rec = {"status": "blocked", "reason": reason, "unblock": unblock, "roster_sha256": sha}
    assert An.withdrawn_legs(b, roster, roster_sha256=sha) == {BFCL: rec}
    v = An.gate_saved_analysis(json.loads(saved.read_text()), roster, roster_path=world["roster"])
    fin = v["final"][BFCL]
    assert fin["source"] == An.FINAL_SOURCE_WITHDRAWN == "withdrawn" and fin["final"] is False and fin["cells"] is None and fin["paired"] is None
    wd = fin["withdrawn"]
    assert (wd["status"], wd["reason"], wd["unblock"], wd["roster_sha256"], wd["kept_from"]) == ("blocked", reason, unblock, sha, "rung1")
    assert wd["cells"] == kept_cells and set(wd["paired"]) == set(kept_paired)                       # the numbers, verbatim, for the record
    keys = ("delta", "ci_low", "ci_high", "b_pass_to_fail", "c_fail_to_pass", "mcnemar_exact_p", "excludes_zero")
    assert all({k: wd["paired"][c][k] for k in keys} == {k: kept_paired[c][k] for k in keys} for c in kept_paired)
    assert v["final"][BCB]["final"] and v["final"][BCB]["source"] == "rung1" and v["final"][BCB]["cells"] == b["final"][BCB]["cells"]   # untouched
    assert v[An.WITHDRAWN_LEGS_KEY] == {BFCL: rec} == v["publication"][An.WITHDRAWN_LEGS_KEY] and v["publication"]["out"] == {}
    assert v["wave"]["legs"][BFCL]["status"] == "blocked" and v["wave"]["legs"][BFCL]["status_at_analysis"] == "runnable" and v["wave"]["legs"][BFCL]["reason"] == reason
    assert BFCL in v["legs"] and BFCL in v["breakdowns"]["bfcl_by_category"]["label"] or "bfcl_by_category" in v["breakdowns"]    # diagnostics / record stay
    assert v["figures"]["withheld"] is True and v["figures"][An.WITHDRAWN_LEGS_KEY] == [BFCL] and "url_base" not in v["figures"]
    nofigs = json.loads(saved.read_text()); nofigs.pop("figures")
    assert "figures" not in An.gate_saved_analysis(nofigs, roster, roster_path=world["roster"])      # nothing to withhold, no marker
    again = An.gate_saved_analysis(v, roster, roster_path=world["roster"])                            # idempotent on the view
    assert again["final"][BFCL]["source"] == "withdrawn" and again[An.WITHDRAWN_LEGS_KEY] == {BFCL: rec}
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
    # the cross-wave summary: one WITHDRAWN row per steered cell on that leg — no Δ, no p, the reason; the other leg reads
    s = An.summarize([saved], roster_path=world["roster"], ledger=world["ledger"])
    wrows = [r for r in s["readings"] if r["task"] == BFCL]
    assert len(wrows) == len(b["steered"]) and all(r["withdrawn"] and r["source"] == "withdrawn" and not r["final"] and r["delta_pp"] is None
                                                     and r["mcnemar_p"] is None and r["reading"].startswith(f"withdrawn — {reason}")
                                                     and "instrument failure: instrument failure" not in r["reading"] for r in wrows)
    assert all(r["delta_pp"] is not None and not r["withdrawn"] for r in s["readings"] if r["task"] == BCB)
    assert s[An.WITHDRAWN_LEGS_KEY] == {"gemma4e4b_main": {BFCL: rec}}
    sc = An.summary_scalars(s)
    assert sc["gemma4e4b_main/bfcl200/diff_lora/delta_pp"] is None and sc["gemma4e4b_main/bfcl200/diff_lora/final_source"] == "withdrawn"
    # W&B: the leg logs its status and NOTHING that reads as a reading — not even under the rung-1 keys
    wb = An.wandb_scalars(v)
    assert wb["bfcl200/final_source"] == "withdrawn" and wb["bfcl200/withdrawn"] == 1 and wb["bfcl200/final"] == 0
    assert not any(k.startswith("bfcl200/") and any(x in k for x in ("pass_rate", "paired", "rung1", "wilson")) for k in wb) and wb["bcb200/paired/diff_lora/delta"] is not None
    # the doc: a clearly labelled WITHDRAWN block quoting the kept numbers with the reason; the headline names it and reads nothing
    md = An.render_results_doc(s["waves"], roster=roster, summary=s)
    wave_md = md.split("## Wave `gemma4e4b_main`")[1].split("## Wave `")[0]
    assert "**WITHDRAWN — instrument failure: BFCL-200 (function calling)**" in wave_md and reason in wave_md and unblock in wave_md
    assert f"{kept_cells[NONE]['k']}/200 at cap {F.CAP_OFF[BFCL]}" in wave_md and "NOT a reading (withdrawn)" in wave_md and "FOR THE RECORD" in wave_md
    assert "kept from the rung1 reading" in wave_md and "legs BCB-200, BFCL-200 (WITHDRAWN)" in wave_md and "| BCB-200, BFCL-200 (WITHDRAWN) |" in md
    headline = wave_md.split("# Headline (final readings)")[1].split("# Decoding protocol")[0]
    assert f"**WITHDRAWN**: {reason}" in headline and "(function calling)** (final:" not in headline and "sandboxed verify)** (final: rung1)" in headline
    assert "instrument failure: instrument failure" not in md                            # the category is not prepended twice
    assert "predates the withdrawal" in wave_md                                            # a render-time-only withdrawal says where the record is NOT
    assert "BFCL-200 by category" not in wave_md and "BFCL discordant items" not in wave_md and "BFCL-200 breakdown and quoted items: not rendered" in wave_md
    assert "BigCodeBench-200 by library family" in wave_md and "BigCodeBench failures, quoted raw" in wave_md
    assert "BFCL-200 (WITHDRAWN) |" in wave_md.split("# Diagnostics")[1]                                # the diagnostics row is labelled, kept
    cross = md.split("## Cross-wave summary")[1].split("## Wave")[0]
    assert any("| BFCL-200 | withdrawn | withdrawn | — | withdrawn — instrument failure" in ln for ln in cross.splitlines()) and "A `withdrawn` row" in cross
    # the paper rows: the leg's rows STAY with `status: withdrawn`, every measurement null, the note naming the withdrawal; no refusal
    rows = An.paper_rows(v)
    wd_rows = [r for r in rows if r["eval_id"] == "bfcl_200" and r["cell_id"] in v["cells"]]
    assert len(wd_rows) == len(v["cells"]) and all(r["status"] == f"withdrawn: {reason}" for r in wd_rows)
    assert all(r[k] is None for r in wd_rows for k in ("value", "k", "n", "ci_lo", "ci_hi", "delta", "delta_lo", "delta_hi", "delta_p", "at_cap_share", "decode_cap_tokens"))
    assert all("WITHDRAWN as a capability reading (D79)" in r["note"] and f"final[{BFCL}].withdrawn" in r["note"] and r["decode"].startswith("benchmark protocol")
               and "WITHDRAWN" in r["decode"] and r["thinking"] == "off" for r in wd_rows)
    assert next(r for r in wd_rows if r["cell_id"] == DIFF_LORA)["object_artifact_sha256"] == SHA[DIFF_LORA]         # identity stays
    assert all(r["status"] == "graded" and r["value"] is not None for r in rows if r["eval_id"] == "bigcodebench_200" and r["cell_id"] in v["cells"])
    assert len(rows) == expected_paper_rows(v)
    pq = tmp_path / "rows.parquet"
    assert An.write_paper_rows(v, pq)["n_added"] == len(rows) and int(pd.read_parquet(pq)["status"].str.startswith("withdrawn").sum()) == len(wd_rows)
    # the figures: the reading figures draw the OTHER leg only, titled with the withdrawal; the BFCL breakdown is the withdrawn placeholder
    fig = An.fig_paired_delta(v)
    ylabels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    blob = _figure_blob(fig)
    assert "BFCL-200 withdrawn" in blob and ylabels and not any(lbl.startswith("BFCL-200") for lbl in ylabels)
    fig = An.fig_pass_rate_by_cell(v)
    xlabels = [t.get_text().split("\n")[0] for t in fig.axes[0].get_xticklabels()]
    assert "BFCL-200 withdrawn" in _figure_blob(fig) and xlabels == [An.TASK_LABEL[BCB]]
    assert "WITHDRAWN" in _figure_blob(An.fig_bfcl_by_category(v))
    assert "BFCL-200 (withdrawn)" in _figure_blob(An.fig_diagnostics(v), ticks=True)
    blob = _figure_blob(An.fig_summary_paired_delta(s))
    assert f"{len(b['steered'])} withdrawn" in blob and "WITHDRAWN (D79; not drawn): gemma4e4b_main BFCL-200" in blob     # one contrast per steered cell
    assert "1 leg(s) withdrawn" in _figure_blob(An.fig_summary_pass_rate(s))
    assert set(An.make_figures(v, tmp_path / "figs" / "wave")) == set(An.FIGURE_STEMS)
    assert set(An.make_figures(s, tmp_path / "figs" / "summary", figures=An.SUMMARY_FIGURES)) == set(An.SUMMARY_FIGURE_STEMS) and plt.get_fignums() == []
    # --analysis + --write-results end to end: the doc carries the block; the gated figures regenerate under the gated namespace
    rc = An.main(["--roster", str(world["roster"]), "--analysis", str(saved), "--write-results", str(tmp_path / "r.md"), "--figs-root", str(tmp_path / "figs2"),
                  "--figs-url-base", "https://example/figs"])
    r_md = (tmp_path / "r.md").read_text()
    assert rc == 0 and "WITHDRAWN — instrument failure" in r_md and f"/gated-{sha[:8]}/final_paired_delta.png" in r_md
    # a FRESH --analyze of stores that file the blocked leg still refuses (a store is never analysed under a label the roster did not launch):
    # the withdrawal is the SAVED analysis's gate, not a new reading
    with pytest.raises(An.AnalysisError, match="legs the roster marks blocked"):
        _analyze(world, wave="gemma4e4b_main", rung1=world["main"])
    # the same saved analysis under a roster that runs the leg reads it again; the file is byte-identical throughout
    write_roster(world["roster"], fixture_roster())
    back = An.gate_saved_analysis(json.loads(saved.read_text()), An.load_roster(world["roster"]), roster_path=world["roster"])
    assert back["final"][BFCL]["final"] and back["final"][BFCL]["source"] == "rung1" and back[An.WITHDRAWN_LEGS_KEY] == {}
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest


def test_a_withdrawn_merged_leg_keeps_the_merged_numbers_and_the_doc_counts_residual_capped_passes(tmp_path):
    """A withdrawn leg whose ladder was APPLIED (the llama BFCL shape: fired, re-run at 16,384, merged): the kept numbers are the
    MERGED ones. The diagnostics name the rung they show, a merged leg ALSO shows its final at-cap counts (cap 16,384), the
    residual capped items the scorer still PASSED are counted from the merged rows (never hand-typed), degeneration reads as not
    measured, and the ladder section says capped outputs keep their benchmark score."""
    import copy
    import matplotlib.pyplot as plt
    w = _merged_main_world(tmp_path, rung2_tree=TREE, admit=False)
    A, lora = w["A"], w["A"].LORA
    a = A._analyze(w["world"], ladder_merge=w["merge_dir"], selection=w["sel"], rung2_dir=w["rung2"])
    assert a["ladder"]["applied"] and a["final"][BCB]["source"] == "merged"
    cell = a["ladder"]["tasks"][BCB]["cells"][lora]
    assert cell["n_at_cap_residual"] == 1 and cell["n_at_cap_residual_pass"] == 0          # the fixture's residual capped item is scored 0 (make_rung2)
    assert An.residual_capped_passes(cell, task=BCB) == 0
    old = {k: v for k, v in cell.items() if k != "n_at_cap_residual_pass"}                  # a record predating the field reads the merged rows it names
    assert An.residual_capped_passes(old, task=BCB) == 0
    assert An.residual_capped_passes({**old, "merged_rows_sha256": "0" * 64}, task=BCB) is None      # a store that is not the record's: not recorded
    assert An.rung1_cap_label(a) == "rung 1 (cap 1,024 / 4,096)"
    md = An.render_wave_doc(a)
    assert "# Diagnostics — rung 1 (cap 1,024 / 4,096): unparseable" in md and An.DEGENERATION_NOTE in md
    assert "Final merged at-cap counts (cap 16,384)" in md and "| 0 of 1 |" in md and "at-cap (stop_reason = max_tokens) — rung 1 |" in md
    assert ("Capped outputs keep their benchmark score (0 of the 1 residual capped diff_lora BCB-200 outputs are scored passes); readings are qualified "
            "by the tested caps") in md
    six = copy.deepcopy(a); six["ladder"]["tasks"][BCB]["cells"][lora].update(n_at_cap_residual=12, n_at_cap_residual_pass=6)
    assert "6 of the 12 residual capped diff_lora BCB-200 outputs are scored passes" in An.render_wave_doc(six)
    blob = _figure_blob(An.fig_diagnostics(a), ticks=True)
    assert "rung 1 (cap 1,024 / 4,096)" in blob and "hatched: merged, cap 16,384" in blob
    # the current roster blocks the merged leg: WITHDRAWN keeps the MERGED numbers; the ladder figure and lines say so
    rp = w["world"]["roster"]
    saved = An.write_analysis(a, w["world"]["rung1"])
    write_roster(rp, _block_leg(fixture_roster(), "gemma4e4b_main", "bigcodebench_200"))
    v = An.gate_saved_analysis(json.loads(saved.read_text()), An.load_roster(rp), roster_path=rp)
    fin = v["final"][BCB]
    assert fin["source"] == "withdrawn" and fin["withdrawn"]["kept_from"] == "merged" and fin["withdrawn"]["cells"] == a["final"][BCB]["cells"]
    assert fin["withdrawn"]["cap_by_cell"][lora] == 16_384 and v["final"][BFCL]["final"]
    md = An.render_wave_doc(v)
    assert "kept from the merged reading" in md and "BCB-200 (WITHDRAWN — record, not a reading)" in md and "WITHDRAWN (D79): not a reading." in md
    assert "BigCodeBench-200 by library family" not in md and "BigCodeBench failures, quoted raw" not in md and "BFCL-200 by category" in md
    assert "BCB-200 WITHDRAWN (D79)" in _figure_blob(An.fig_ladder(v)) and "WITHDRAWN" in _figure_blob(An.fig_bcb_by_family(v))
    assert set(An.make_figures(v, tmp_path / "figs")) == set(An.FIGURE_STEMS) and plt.get_fignums() == []


def test_the_reading_rule_is_the_exact_test_and_every_emphasis_follows_it_not_the_interval():
    """The rule in code (exact McNemar p < 0.05 AND Δ < 0) is right and stays; the figures highlighted on the bootstrap
    `excludes_zero` instead and disagreed on ONE row — gemma-it si_safety BFCL: −2.0 [−4.0, −0.5], b 4 / c 0, p 0.125 (four one-way
    discordants give the percentile bootstrap ≈ 1.8 % mass at zero). Now: highlight iff a cost is detected by the exact test; the
    interval is drawn as an estimation interval; where the two disagree ONE sentence explains sparse discordance."""
    import matplotlib.pyplot as plt
    p = _paired(4, 0, lo=-0.04, hi=-0.005)                                            # the review's row
    assert p["excludes_zero"] is True and p["cost_detected"] is False and An.cost_detected(p) is False and p["mcnemar_exact_p"] == pytest.approx(0.125)
    assert p["reading"].startswith("no cost detected (-2.0 pp [-4.0 pp, -0.5 pp], p = 0.125)") and "cost detected:" not in p["reading"]
    assert An.interval_disagrees(p) and "sparse discordance" in p["reading"] and "b + c = 4" in p["reading"] and "the exact test is the rule" in p["reading"]
    assert An.zero_mass_share(p) == pytest.approx((196 / 200) ** 200) and "((200 − 4)/200)^200 = 1.76 %" in p["reading"]
    assert "fewer than the 2.5 %" in p["reading"] and "coverage" not in p["reading"] and "cannot place" not in p["reading"]
    q = _paired(15, 5, lo=-0.095, hi=-0.01)                                            # a detected cost: p 0.0414, Δ −5 pp
    assert q["cost_detected"] is True and q["reading"].startswith("cost detected: -5.0 pp [-9.5 pp, -1.0 pp], exact McNemar p = 0.0414") and not An.interval_disagrees(q)
    r = _paired(15, 5, lo=-0.09, hi=0.001)                                             # detected by the test while the interval touches 0: the sentence rides too
    assert r["cost_detected"] is True and An.interval_disagrees(r) and "interval includes 0 while the exact test rejects" in r["reading"]
    g = _paired(0, 4, lo=0.005, hi=0.04)                                               # a positive estimate is never a cost, whatever the interval
    assert g["cost_detected"] is False and not An.interval_disagrees(g) and g["reading"].startswith("no cost detected")
    fig, ax = plt.subplots()
    An._forest(ax, [("sparse", p, An.CAT["blue"]), ("cost", q, An.CAT["blue"]), ("gain", g, An.CAT["blue"])])
    assert [cont[0].get_color() for cont in ax.containers] == [An.MUTED, An.CAT["blue"], An.MUTED]      # coloured iff cost detected
    labels = [t.get_text() for t in ax.texts]
    assert any("p = 0.125 — interval excludes 0 (sparse discordance; the test is the rule)" in t for t in labels) and any("p = 0.0414" in t for t in labels)
    plt.close(fig)
    # a fresh paired section records the bit beside the interval
    a_vals = {str(i): 1.0 for i in range(200)}
    b_vals = {**a_vals, **{str(i): 0.0 for i in range(4)}}
    ps = An.paired_section(a_vals, b_vals)
    assert ps["b_pass_to_fail"] == 4 and ps["c_fail_to_pass"] == 0 and ps["cost_detected"] is False and ps["excludes_zero"] is True
    assert "sparse discordance" in An.reading_rule(ps) and An.reading_rule(ps).startswith("no cost detected")


def test_p_values_never_print_as_zero():
    """8.7e-6 and 3.9e-4 printed as `0.000`; now `< 0.001`, else three significant figures."""
    assert An._p_value(8.7e-6) == "< 0.001" == An._p_value(3.9e-4) and An._p(8.7e-6) == "p < 0.001"
    assert An._p_value(0.0413) == "0.0413" and An._p_value(0.00432) == "0.00432" and An._p_value(0.125) == "0.125"
    assert An._p_value(0.5) == "0.500" and An._p_value(1.0) == "1.000" and An._p(0.125) == "p = 0.125" and An._p_value(None) == "n/a"
    p = _paired(44, 11)
    assert "exact McNemar p < 0.001" in p["reading"] and "0.000" not in p["reading"]


def test_the_paper_thinking_column_is_the_mode_string_not_the_producers_dict(world):
    """The producer records thinking as {"mode": "off", "render": "native"} and the column carried that dict's repr —
    a notebook filter `thinking == "off"` returned 0 rows. The column is the MODE ("off"); the render detail is not in the schema."""
    assert An.thinking_mode_of({"mode": "off", "render": "native"}) == "off" == An.thinking_mode_of("off") == An.thinking_mode_of(None)
    a = _analyze(world)
    a["identity"]["thinking"] = {"mode": "off", "render": "native"}                 # every real analysis carries this shape
    rows = An.paper_rows(a)
    assert {r["thinking"] for r in rows} == {"off"} and len(rows) == expected_paper_rows(a)


def test_the_reading_strings_of_a_saved_analysis_are_rederived_under_the_current_rule(world):
    """The `reading` text is stored in analysis.json at analysis time; the one selector re-derives it from the numbers, so a saved
    record renders the current rule (p format, the sparse-discordance sentence, the decision bit) without being rewritten."""
    a = _analyze(world)
    raw = json.loads(json.dumps(a))
    for leg in raw["legs"].values():
        for p in leg["paired"].values():
            p["reading"] = "STALE reading text (p = 0.000)"; p.pop("cost_detected", None)
    fin = An.final_readings(raw)
    for task, f in fin.items():
        if f["final"]:
            assert all(p["reading"].startswith(("no cost detected", "cost detected")) and "STALE" not in p["reading"] and "cost_detected" in p for p in f["paired"].values())
    assert "STALE" not in An.render_wave_doc(raw) and "0.000" not in An.render_wave_doc(raw)


def test_the_roster_figures_separate_eligibility_from_result(world, tmp_path):
    """The roster figure conflated eligibility with result (the provisional mark lost, pending drawn as measured,
    blocked legs absent). Now: fill = eligibility (runnable / queued / blocked, per entry × leg), marker = result (measured /
    provisional / pending / withdrawn / out / not run), on a cell × leg grid — per wave and across the lane."""
    import matplotlib.pyplot as plt
    a = _analyze(world)                                # it wave: BCB fired (pending), BFCL final; the constitution objects are batch-8 provisional
    it = {label.split(" (")[0]: per_leg for label, _, per_leg in An.roster_grid_items(a)}
    assert it["none"] == {BCB: ("runnable", "pending"), BFCL: ("runnable", "measured")}
    assert it["const_lora"] == {BCB: ("runnable", "pending"), BFCL: ("runnable", "provisional")} == it["const_psr"]
    assert it["si_safety"][BFCL] == ("runnable", "measured") and it["ctl_lora"] == {BCB: ("queued", "not run"), BFCL: ("queued", "not run")}
    assert it["alphasteer_as_fit"] == {BCB: ("queued", "not run"), BFCL: ("queued", "not run")}
    # a wave with a non-runnable leg: eligibility carries the leg's status on that leg, not run (the qwen BFCL leg is QUEUED on
    # the LIVE half of the S5 probe — it read `blocked` before the qwen3_xml pin)
    q = _analyze(world, wave="qwen35_4b_main", rung1=make_wave(tmp_path / SUFFIX_QWEN, world["doc"], "qwen35_4b_main", roster_path=world["roster"], run_id=RUN_QWEN))
    qi = {label.split(" (")[0]: per_leg for label, _, per_leg in An.roster_grid_items(q)}
    assert qi["none"] == {BCB: ("runnable", "measured"), BFCL: ("queued", "not run")} and qi["diff_lora"] == {BCB: ("runnable", "measured"), BFCL: ("queued", "not run")}
    # an OUT cell (PSR blocked by the shipped roster) and a WITHDRAWN leg (the it wave's BFCL blocked), together
    saved = An.write_analysis(a, world["it"])
    write_roster(world["roster"], _block_leg(roster_doc(), "gemma4e4b_it", "bfcl_200"))
    v = An.gate_saved_analysis(json.loads(saved.read_text()), An.load_roster(world["roster"]), roster_path=world["roster"])
    vi = {label.split(" (")[0]: per_leg for label, _, per_leg in An.roster_grid_items(v)}
    assert vi["const_psr"] == {BCB: ("blocked", "out"), BFCL: ("blocked", "out")} and vi["none"] == {BCB: ("runnable", "pending"), BFCL: ("blocked", "withdrawn")}
    assert vi["const_lora"] == {BCB: ("runnable", "pending"), BFCL: ("blocked", "withdrawn")} and vi["ctl_psr"] == {BCB: ("blocked", "not run"), BFCL: ("blocked", "not run")}
    fig = An.fig_roster_status(v)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    xlabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    texts = list(_figure_texts(fig))
    title = next(t.replace("\n", " ") for t in texts if t.startswith("Roster —"))
    blob = _figure_blob(fig)
    assert "2 out" in title and "withdrawn" in title and "pending" in title and "provisional" not in title   # out cells are not provisional results
    assert An.PSR_OUT_LABEL in blob and any(t.startswith(An.PSR_OUT_LABEL) for t in texts)
    assert {"eligibility: runnable", "eligibility: queued", "eligibility: blocked", "result: measured", "result: provisional", "result: pending",
            "result: withdrawn", "result: out"} <= set(legend)
    assert xlabels == [An.TASK_SHORT[BCB], An.TASK_SHORT[BFCL]]
    # across the lane: the analysed wave's grid plus every unanalysed wave's entries with the roster's leg status (a non-runnable leg
    # shows its status — the qwen BFCL leg is QUEUED on the live probe; the llama legs stay blocked)
    s = An.summarize([saved], roster_path=world["roster"], ledger=world["ledger"])
    assert s["unanalysed_wave_legs"]["qwen35_4b_main"][BFCL]["status"] == "queued" and s["unanalysed_wave_legs"]["gemma4e4b_main"][BFCL]["status"] == "runnable"
    assert s["unanalysed_wave_legs"]["llama8b_main"][BFCL]["status"] == "blocked"
    blob = _figure_blob(An.fig_summary_roster(s))
    assert "Roster across the lane" in blob and "withdrawn" in blob and "not run" in blob
    assert plt.get_fignums() == []


def test_analyze_with_a_launch_roster_persists_the_withdrawal_and_draws_no_withdrawn_leg_at_the_root(world, tmp_path):
    """The withdrawal used to live only in the render-time gated view — `load_run` refuses stores that file a blocked leg, so
    the llama waves were analysed under the roster that LAUNCHED them and their saved analysis.json, root figures and W&B runs still read
    BFCL as a final merged reading. `--launch-roster` admits the stores under the launch roster (a store is still never analysed under a
    label the roster did not launch) while `--roster` (the current one) gates: ONE canonical record with `withdrawn_legs` and
    `final[leg].withdrawn` persisted, root figures drawn from the gated record, W&B scalars from it, both rosters in `publication`."""
    import matplotlib.pyplot as plt
    launch = world["roster"]                                            # the fixture stores' launch roster: both legs runnable
    reason = "instrument failure (D79): the template renders no tools"
    current = write_roster(tmp_path / "lane" / "current.yaml", _block_leg(fixture_roster(), "gemma4e4b_main", "bfcl_200", reason=reason))
    kw = dict(wave_name="gemma4e4b_main", repo_root=world["repo"], ledger=world["ledger"], sandbox_state_root=world["tmp"] / "sandbox-state")
    sha_cur, sha_launch = W.roster_sha256(current), W.roster_sha256(launch)
    with pytest.raises(An.AnalysisError, match="legs the roster marks blocked"):            # without the launch roster: unchanged, refuses
        An.analyze(world["main"], roster_path=current, **kw)
    a = An.analyze(world["main"], roster_path=current, launch_roster_path=launch, **kw)
    assert a[An.WITHDRAWN_LEGS_KEY] == {BFCL: {"status": "blocked", "reason": reason, "unblock": "a tool-aware template verified in-lane, then the cells re-run",
                                                "roster_sha256": sha_cur}}
    fin = a["final"][BFCL]
    assert fin["source"] == "withdrawn" and fin["final"] is False and fin["cells"] is None and fin["withdrawn"]["kept_from"] == "rung1"
    assert fin["withdrawn"]["cells"][NONE]["n"] == 200 and set(fin["withdrawn"]["paired"]) == set(a["steered"])
    assert a["final"][BCB]["final"] and a["tasks"] == [BCB, BFCL] and BFCL in a["legs"]      # the stores were admitted under the launch roster
    assert (a["publication"]["roster_sha256"], a["publication"]["launch_roster_sha256"], a["publication"]["launch_roster"]) == (sha_cur, sha_launch, str(launch))
    assert (a["wave"]["roster_sha256"], a["wave"]["launch_roster_sha256"]) == (sha_cur, sha_launch)
    assert a["wave"]["legs"][BFCL]["status"] == "blocked" and a["wave"]["legs"][BFCL]["status_at_analysis"] == "runnable"
    assert str(world["main"]) in An.withdrawn_record_location(a) and "predates the withdrawal" not in An.withdrawn_record_location(a)
    # the SAVED record carries it; a re-gate under the current roster is idempotent and keeps the launch fields
    saved_path = An.write_analysis(a, world["main"])
    raw = json.loads(saved_path.read_text())
    assert raw[An.WITHDRAWN_LEGS_KEY] == a[An.WITHDRAWN_LEGS_KEY] and raw["final"][BFCL]["source"] == "withdrawn" and raw["publication"]["launch_roster_sha256"] == sha_launch
    v = An.gate_saved_analysis(raw, An.load_roster(current), roster_path=current)
    assert v["final"][BFCL]["source"] == "withdrawn" and v["publication"]["launch_roster_sha256"] == sha_launch and v["publication"]["out"] == {}
    # the same file as both rosters: behaviour unchanged (nothing withdrawn; the launch fields still recorded)
    b = An.analyze(world["main"], roster_path=launch, launch_roster_path=launch, **kw)
    assert b[An.WITHDRAWN_LEGS_KEY] == {} and b["final"][BFCL]["final"] and b["publication"]["launch_roster_sha256"] == sha_launch
    # rosters that disagree on the wave's identity refuse before any store is read
    other = fixture_roster(); other["waves"]["gemma4e4b_main"]["lane_kind"] = "chat-ckpt"
    with pytest.raises(An.AnalysisError, match="disagree on wave 'gemma4e4b_main'"):
        An.analyze(world["main"], roster_path=write_roster(tmp_path / "lane" / "other.yaml", other), launch_roster_path=launch, **kw)
    # the CLI: --analyze --launch-roster writes the withdrawal into analysis.json and draws the ROOT figures from the gated record
    rc = An.main(["--roster", str(current), "--launch-roster", str(launch), "--wave", "gemma4e4b_main", "--analyze", str(world["main"]),
                  "--repo-root", str(world["repo"]), "--ledger", str(world["ledger"]), "--no-wandb", "--figs-root", str(tmp_path / "figs"),
                  "--figs-url-base", "https://example/figs", "--write-results", str(tmp_path / "r.md")])
    assert rc == 0
    saved = json.loads((world["main"] / An.ANALYSIS_FILENAME).read_text())
    assert saved["final"][BFCL]["source"] == "withdrawn" and saved["figures"]["url_base"] == "https://example/figs/" + saved["run_name"]
    assert saved["figures"]["gated"] == {"roster_sha256": sha_cur, "out": [], An.WITHDRAWN_LEGS_KEY: [BFCL], "at_root": True}
    root = tmp_path / "figs" / saved["run_name"]
    assert (root / "final_paired_delta.png").is_file() and not (root / f"gated-{sha_cur[:8]}").exists()
    fig = An.fig_pass_rate_by_cell(saved)
    xl = [t.get_text().split("\n")[0] for t in fig.axes[0].get_xticklabels()]
    assert xl == [An.TASK_LABEL[BCB]] and "BFCL-200 withdrawn" in _figure_blob(fig)             # the root figures draw no withdrawn leg
    assert "WITHDRAWN" in _figure_blob(An.fig_bfcl_by_category(saved))
    md = (tmp_path / "r.md").read_text()
    assert "Figures drawn from the GATED record under the current roster" in md and f"https://example/figs/{saved['run_name']}/final_paired_delta.png" in md
    assert f"`{world['main']}/analysis.json` (analysed with `--launch-roster` {sha_launch[:8]} under the current roster {sha_cur[:8]})" in md
    assert "predates the withdrawal" not in md and "**WITHDRAWN — instrument failure: BFCL-200 (function calling)**" in md
    # W&B scalars of the persisted record (unit level, no network): the leg's status only, no withdrawn rate / Δ / p, not even under rung1
    wb = An.wandb_scalars(saved)
    assert wb["bfcl200/final_source"] == "withdrawn" and wb["bfcl200/withdrawn"] == 1 and wb["bfcl200/final"] == 0
    assert not any(k.startswith("bfcl200/") and any(x in k for x in ("pass_rate", "paired", "rung1", "wilson")) for k in wb) and wb["bcb200/final"] == 1
    # --analysis on the saved record re-gates idempotently: the root figures stay linked, no gated-<sha8> regeneration
    rc = An.main(["--roster", str(current), "--analysis", str(world["main"] / An.ANALYSIS_FILENAME), "--write-results", str(tmp_path / "r2.md"),
                  "--figs-root", str(tmp_path / "figs"), "--figs-url-base", "https://example/figs"])
    assert rc == 0 and not (root / f"gated-{sha_cur[:8]}").exists() and f"https://example/figs/{saved['run_name']}/final_paired_delta.png" in (tmp_path / "r2.md").read_text()
    # the paper note names the saved file as the holder of final[<leg>].withdrawn
    wd_rows = [r for r in An.paper_rows(saved) if r["eval_id"] == "bfcl_200" and r["cell_id"] in saved["cells"]]
    assert wd_rows and all(f"{world['main']}/analysis.json" in r["note"] and "predates" not in r["note"] and r["status"] == f"withdrawn: {reason}" for r in wd_rows)
    assert An._withdrawn_reason_text("instrument failure (D79): x") == "instrument failure (D79): x" and An._withdrawn_reason_text("parser unverified") == "instrument failure: parser unverified"
    assert plt.get_fignums() == []



def test_ladder_spec_falls_back_to_the_certificates_recorded_ladder_when_the_table_refuses(monkeypatch):
    """The landed tree keeps main's geometry rows, under which llama8b's row is provisional and `rung_probe.ladder_spec`
    refuses; the llama ladders ran under a measured row and their certificates record the ladder walked — the analysis judges
    the wave record by THAT ladder. Table OK → the table's spec; table refuses + one recorded ladder → the recorded one; table
    refuses + disagreeing certificates → refuse."""
    from sisteer.capeval.tinycap import rung_probe as RP
    lad = {"model_key": "llama8b", "window": 32768, "seqs": [12], "identity_S": 12}
    certs = {"a": {"ladder": dict(lad)}, "b": {"ladder": dict(lad)}}
    spec, source = An.ladder_spec_for_certificates("gemma4e4b", certs)          # a measured row: the table sizes it
    assert source == "table" and spec.model_key == "gemma4e4b"

    def refuse(model_key, *, window=RP.LADDER_WINDOW):
        raise RP.LadderError(f"the ladder needs the MEASURED budget row of {model_key!r}; the geometry table holds 'provisional-conservative'")
    monkeypatch.setattr(RP, "ladder_spec", refuse)
    spec, source = An.ladder_spec_for_certificates("llama8b", certs)
    assert (spec.window, spec.seqs, spec.identity_S, spec.model_key) == (32768, (12,), 12, "llama8b")
    assert source.startswith("certificates (the shared table refuses:")
    other = {"a": {"ladder": dict(lad)}, "b": {"ladder": dict(lad, seqs=[24, 22])}}
    with pytest.raises(An.AnalysisError, match="2 distinct `ladder` blocks"):
        An.ladder_spec_for_certificates("llama8b", other)
    with pytest.raises(An.AnalysisError, match="names model"):
        An.ladder_spec_for_certificates("qwen35_4b", certs)


# --------------------------------------------------------------------------- #
# held out of pooled category means                                           #
# --------------------------------------------------------------------------- #


def test_a_held_out_entry_is_reported_with_its_numbers_and_stamped_out_of_pooled_means(world, tmp_path):
    """A cell that refused all 200 coding items (the qwen35_4b it LoRA subtraction): REPORT the reading — do not withdraw it —
    but hold it OUT of every pooled category mean. The declaration is the roster entry's
    `pooling: {held_out: true, ruled_by, reason}` block, a PUBLICATION decision read by the gate from the CURRENT roster (never the
    launch-time record); the gated view stamps `publication.held_out`, the paper rows carry a `held_out` column (True on that cell's
    measured rows, False everywhere else) with the declaration and the reason LEADING the note, the summary readings and the results doc
    name it, and `status` stays `graded` with the value and the paired delta intact. Withdrawing the declaration clears the stamp."""
    import pandas as pd
    a = _analyze(world)
    assert a["publication"][An.HELD_OUT_KEY] == {} and all(r["held_out"] is False for r in An.paper_rows(a))
    saved = An.write_analysis(a, world["it"])
    doc = fixture_roster()
    blk = {"held_out": True, "ruled_by": "the paper owner, 2026-09-13 (test)",
           "reason": "the arm refuses every item; e.g. \"I can't help with creating code designed to bypass security controls\""}
    doc["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["pooling"] = blk
    write_roster(world["roster"], doc)
    held = An.load_roster(world["roster"])
    v = An.gate_saved_analysis(json.loads(saved.read_text()), held, roster_path=world["roster"])
    assert v["publication"][An.HELD_OUT_KEY] == {CONST_LORA: blk} and v["publication"]["out"] == {} and CONST_LORA in v["cells"]
    assert next(e for e in v["wave"]["entries"] if e["cell_id"] == CONST_LORA)["pooling"] == blk
    rows = An.paper_rows(v)
    assert An.PAPER_COLUMNS[-1] == "held_out" and all(tuple(r) == An.PAPER_COLUMNS for r in rows)
    mine = [r for r in rows if r["cell_id"] == CONST_LORA and r["status"] in ("graded", "pending ladder")]
    assert mine and all(r["held_out"] is True and r["note"].startswith("HELD OUT of pooled category means (the paper owner")
                        and "refuses every item" in r["note"] and "held_out=true" in r["provenance"] for r in mine)
    g = next(r for r in mine if r["status"] == "graded")
    assert g["value"] is not None and g["delta"] is not None and g["status"] == "graded"          # reported, not withdrawn
    assert all(r["held_out"] is False for r in rows if r["cell_id"] != CONST_LORA)
    s = An.summarize([saved], roster_path=world["roster"], ledger=world["ledger"])
    rd = [r for r in s["readings"] if r["cell"] == CONST_LORA]
    assert rd and all(r["held_out"] and not r["out"] and not r["withdrawn"] for r in rd)
    assert all(not r["held_out"] for r in s["readings"] if r["cell"] != CONST_LORA)
    sc = An.summary_scalars(s)
    assert sc["gemma4e4b_it/bfcl200/const_lora/held_out"] is True and sc["gemma4e4b_it/bfcl200/const_psr/held_out"] is False   # the fixture runs PSR
    md = An.render_results_doc(s["waves"], roster=held, summary=s)
    assert "Reported, HELD OUT of pooled category means" in md and "refuses every item" in md and "48 columns" in md
    # the cross-wave table labels the arm "(held out)" beside its other marks (the fixture's const_lora is a batch-8 provisional object)
    import re
    assert re.search(r"\| const_lora[^|]*\(held out\) \|", md) and not re.search(r"\| const_psr[^|]*\(held out\)", md)
    pq = tmp_path / "rows.parquet"
    An.write_paper_rows(v, pq)
    df = pd.read_parquet(pq)
    assert str(df["held_out"].dtype) == "bool" and set(df.loc[df["cell_id"] == CONST_LORA, "held_out"]) == {True}
    assert not df.loc[df["cell_id"] != CONST_LORA, "held_out"].any()
    # idempotent on the view; withdrawing the declaration (a roster without the block) clears the stamp and the flag
    assert An.gate_saved_analysis(v, held)["publication"][An.HELD_OUT_KEY] == {CONST_LORA: blk}
    write_roster(world["roster"], fixture_roster())
    v2 = An.gate_saved_analysis(v, An.load_roster(world["roster"]), roster_path=world["roster"])
    assert v2["publication"][An.HELD_OUT_KEY] == {} and all(r["held_out"] is False for r in An.paper_rows(v2))
    assert next(e for e in v2["wave"]["entries"] if e["cell_id"] == CONST_LORA)["pooling"] is None
