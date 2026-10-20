#!/usr/bin/env python3
"""Grading as a CONTINUOUS consumer of the registry's pending-judgment queue — the loop
`reproduce/08_grade.sh --daemon` runs from the first eval submission on, never as a trailing stage.

Every tick (default 120 s): the pending set through the shared worklist — `worklist.RegistryWorklist(model_ids=<the model's ids>,
campaign=…).pending()`, the drivers' own reader over `pending.pending(conn, model_id=…)` (`model_id` and `campaign` are `FILTER_COLUMNS`;
`registry pending-judgment` has no `--model` flag and is not used), every row located on this host as a driver would see it — grouped by
eval FAMILY (the row's `eval_category`, FAMILY_OF_CATEGORY) × the job's model-key spelling (`run.view_legacy`) × the driver's ROOT and
config the group's stores name (FAMILY_LAYOUT, below); each group's store mirrored from its cluster onto the grading host (rsync over the
cluster's ssh alias into the store map's local root) and the family's judging driver called with EVERY argument it requires —
`--worklist registry --run-id <id> … --store-root-map <substrate>=<remote root>=<local root>` plus its root (`--private-root` /
`--results-root` / `--controls-root` / `--out-root`), its `--config` / `--manifest`, `--target`, `--attacks` and `--grades-root` — so it
claims (`pending.claim`) exactly the selected runs before its first judge call and records through `hooks.judgment_scope`; then
`scripts/registry/emit_contrasts.py` files a `run_contrast` for every steered run whose anchor is complete. Rule-scored families (MMLU,
GSM8K) and the Modal-scored ones (BigCodeBench, BFCL) are confirmed, never dispatched. A row another live session has claimed is left alone.

The drivers' required arguments are DERIVED, never guessed (the templates once passed the worklist flags alone and every static /
padding / controls / agentic dispatch died in argparse each tick, so a normal first wave was never judged):
  * the ROOT from the group's mapped store paths through the family's layout (`FAMILY_LAYOUT`: how many components sit between the
    driver's root and a run's store — `<private-root>/<private_run>/<attack>/<cell>` for the sampled, padded and multi-turn lanes,
    `<private-root>/<private_run>/<cell>` for BoN, `<results-root>/<family>/<cell>` for the agentic judge, `<controls-root>/<target>/cells/
    <store>` for the over-refusal judge, `<out-root>/cells/<key>` for Petri) — a group whose stores name two roots, or a run with no store
    path on this host, is refused by name and left pending;
  * the CONFIG from the store's `private_run` (`config_for`): the sampled / multi-turn / BoN lanes name their config after the private run
    (`configs/<private_run>.yaml`), the padded lane through the generator's spelling (`configs/sdfbase/static_padding/sdfbase_full22_<key>
    .padstatic[_<family>]_p<pad>.yaml` for `pad_static_<key>[_<family>]_p<pad>`), Petri through the model key's campaign manifest
    (`configs/capeval/manifests/petri_full22_<model_key>.yaml`) — under `--configs-root` (the front door passes the repo's `configs/`);
    a config that does not exist is refused by name (`config_missing`), never dispatched into an argparse or identity failure;
  * the TARGET from the same private run (`target_for`: the TAG a config's `private_run` carries, which the graders hold `--target` to).

Family → driver (the templates below; a site overrides one with `--driver FAMILY=<template>`; `--extra FAMILY=<args>` appends
further arguments; both may use every placeholder):
  static     scripts/sdfbase/full22/sampled/grade_static_sampled.py       --target --config --private-root --grades-root --attacks
  padding    scripts/sdfbase/full22/static_padding/grade_padded_static.py  the same, ONE (level, attack) per invocation
  bon        scripts/sdfbase/seedrep/grade_full22_native.py --attacks bon --rubric v2 --require-engine lora_native  + the same roots
  multiturn  scripts/sdfbase/seedrep/grade_full22_native.py --attacks <the pending attacks> --require-engine lora_native  + the same roots
  petri      scripts/petri_run.py --manifest <the model key's manifest> analyze --out-root
  agentic    scripts/sdfbase/full22/misalign/misalign_judge.py --results-root --no-escalate
  controls   scripts/served_capability_judge.py --controls-root
Placeholders a template may use: {python} {repo} {registry_root} {run_id_flags} {store_map_flags} {model_id} {model_key} {campaign}
{attacks} {target} {config} {store_root} {private_run} {grades_root} {model_identity_flags} {served_identity_dir} {extra}. A template that
names a placeholder the group cannot
supply refuses that group by name (a `dispatch_refused` event carrying the run ids) instead of running a driver without it. Every command
is printed before it runs; `--dry-run` prints and dispatches nothing. Events go to `<state-dir>/grade_daemon.jsonl`. The API coordination
line is printed at start; the daemon never waits on it.

Two HOLDS stop all work, both re-read every tick and at every Petri start decision: the operator's `<state-dir>/dispatch.hold`
and the SCHEMA hold -- the live registry's migration version or shape is not the code's (`db.require_shape`
over a read-only connection: the writers' own refusal, `SchemaDrift`, observed without migrating or writing anything). While the schema differs
the tick does nothing but the check and its summary line (`held: true, hold: schema`; one `schema_hold` event per distinct gap, one
`schema_hold_released` when the live schema reaches the code's -- by itself, the daemon keeps ticking), the Petri thread starts no drive (one
`petri_drive_held` per tag, under either hold), and a grader whose own claim refuses the schema is held transient (`dispatch_schema_held`: left
pending, copy kept, never quarantined, never counted toward the retryable cap). The migration of 2026-10-02 20:53-21:30Z landed under a running
daemon: the tick read the pending rows (read-only, no version check), mirrored and dispatched, the graders' claims raised
SchemaDrift(have=21, want=22), the classifier read a PERMANENT failure and quarantined the runs, and the completed migration changed none of
the quarantine's release markers (code sha, store generation, the release file) -- legitimate pending grading stayed stuck until a sha change.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from sisteer import coordination  # noqa: E402
from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import harvest as RH  # noqa: E402  -- records_dir: the registry spool's root, for the per-dispatch judgment-record read (the cap ruling)
from sisteer.registry import identity as RI  # noqa: E402  -- run_dir_name: a run's spool directory
from sisteer.registry import migrations as RM  # noqa: E402
from sisteer.registry import pending as P  # noqa: E402
from sisteer.registry import worklist as RW  # noqa: E402
from sisteer.registry.records import utc_now  # noqa: E402

import emit_contrasts as EC  # noqa: E402  -- the sibling script beside this file; a checkout without it is broken and refuses at start
from sisteer import front_door_views as FV  # noqa: E402  -- the models table's classification for a tag-holding caller (demo-import M3)

DRIVER_ID = "scripts/registry/grade_daemon.py"
#: eval CATEGORY (vocab.EVAL_CATEGORIES, a column of every v_pending_judgment row) → FAMILY, the driver table's key. The registry owns the
#: classification, so an eval added to the seeds routes itself; `--eval-family EVAL_ID=FAMILY` overrides one eval id. A category the table
#: does not name (`misalignment`, `omit_params`: no eval carries them) is `unknown`, logged and left pending -- never guessed.
FAMILY_OF_CATEGORY: Dict[str, str] = {"static_attack": "static", "multi_turn": "multiturn", "best_of_n": "bon", "over_refusal": "controls",
                                      "petri": "petri", "agentic": "agentic", "em_organism": "agentic", "capability": "capability"}
#: families the daemon never dispatches: the capability evals are rule-scored in the job (MMLU, GSM8K, ...: the job emits run_metric at
#: generated) or scored inside the Modal sandbox (BigCodeBench, BFCL); `mmlu` / `gsm8k` / `bcb_bfcl` stay accepted --eval-family spellings
CONFIRM_ONLY = ("capability", "mmlu", "gsm8k", "bcb_bfcl")
#: FAMILY → (store depth, config kind): the depth is the number of path components between the driver's ROOT argument and a run's store
#: (the drivers' own layouts, cited in the module docstring); the config kind names how `config_for`
#: resolves the driver's config: from the store's private run (`private_run`), the padded generator's spelling (`padding`), the model key's
#: Petri campaign manifest (`petri_manifest`), or none (the judge needs no config).
FAMILY_LAYOUT: Dict[str, Tuple[int, Optional[str]]] = {
    "static": (3, "private_run"),        # grade_static_sampled:  <private-root>/<private_run>/<attack>/<cell>   (jailbreak_run._gated_defense_root)
    "padding": (3, "padding"),           # grade_padded_static:   <private-root>/pad_static_<key>[_<fam>]_p<pad>/<attack>/<cell>
    "multiturn": (3, "private_run"),     # grade_full22_native:   <private-root>/<private_run>/<attack>/<cell>
    "bon": (2, "private_run"),           # grade_full22_native:   <private-root>/<private_run>/<cell>   (or its bon_responses.jsonl)
    "agentic": (2, None),                # misalign_judge:        <results-root>/<family>/<cell>
    "controls": (3, None),               # served_capability_judge: <controls-root>/<target>/cells/<store>
    "petri": (2, "petri_manifest"),      # petri_run analyze:     <out-root>/cells/<petri_run_key>   (capeval.store.CellStore)
}
#: families whose driver grades ONE eval per invocation (grade_padded_static refuses a multi-family --attacks): grouped per eval id too
ONE_EVAL_PER_DISPATCH = ("padding",)
DEFAULT_TEMPLATES: Dict[str, str] = {
    "static": "{python} {repo}/scripts/sdfbase/full22/sampled/grade_static_sampled.py --target {target} --config {config} --private-root {store_root} --grades-root {grades_root} --attacks {attacks} --worklist registry --registry-root {registry_root} {run_id_flags} {store_map_flags} {extra}",
    "padding": "{python} {repo}/scripts/sdfbase/full22/static_padding/grade_padded_static.py --target {target} --config {config} --private-root {store_root} --grades-root {grades_root} --attacks {attacks} --worklist registry --registry-root {registry_root} {run_id_flags} {store_map_flags} {extra}",
    "bon": "{python} {repo}/scripts/sdfbase/seedrep/grade_full22_native.py --target {target} --config {config} --private-root {store_root} --grades-root {grades_root} --attacks bon --rubric v2 --require-engine lora_native --worklist registry --registry-root {registry_root} {run_id_flags} {store_map_flags} {extra}",
    "multiturn": "{python} {repo}/scripts/sdfbase/seedrep/grade_full22_native.py --target {target} --config {config} --private-root {store_root} --grades-root {grades_root} --attacks {attacks} --require-engine lora_native --worklist registry --registry-root {registry_root} {run_id_flags} {store_map_flags} {extra}",
    "petri": "{python} {repo}/scripts/petri_run.py --manifest {config} analyze --out-root {store_root} --worklist registry --registry-root {registry_root} {run_id_flags} {store_map_flags} {extra}",
    "agentic": "{python} {repo}/scripts/sdfbase/full22/misalign/misalign_judge.py --results-root {store_root} --no-escalate --worklist registry --registry-root {registry_root} {run_id_flags} {store_map_flags} {model_identity_flags} --served-identity-dir {served_identity_dir} {extra}",
    "controls": "{python} {repo}/scripts/served_capability_judge.py --controls-root {store_root} --worklist registry --registry-root {registry_root} {run_id_flags} {store_map_flags} {extra}",
}
RSYNC_FLAGS = ("-rlt", "--no-owner", "--no-group", "--no-perms", "--partial")
#: The files a family's grader reads from the judged ROOT (above the per-run store paths the mirror
#: copies) -- the misalign rows step's SERVED model identity (sisteer.capeval.misalign.rows.resolve_model_identity's candidates, in its
#: order) -- mirrored beside the stores so the grader reads the identity exactly as on the cluster; the registry's identity is only a
#: cross-check there (misalign_judge.cross_check_identity), never an override. Relative to the family's root (store_root_of).
#: The grader never READS them from the mirror root, which is shared -- every dispatch of such a
#: family gets its own immutable snapshot of them (snapshot_served_identity, beside the dispatch log, `--served-identity-dir`), taken under
#: the root's identity lock in the same locked section as the transfer (mirror_root_identity).
ROOT_IDENTITY_FILES: Dict[str, Tuple[str, ...]] = {
    "agentic": ("cells_full.json", "endpoint.snapshot.json", "serving.json", "serving/endpoint.recorded.json"),
}
#: the sampled / multi-turn / BoN private runs: `sdfbase_full22_<TAG>[_sampled_t<T>[_cap<N>]]` -- the TAG is what `--target` takes
#: (grade_static_sampled.RUN_RE for the sampled form; f22_resolve_config's `configs/sdfbase_full22_<TAG>.yaml` for the wave form)
_FULL22_RUN_RE = re.compile(r"^sdfbase_full22_(?P<tag>[a-z0-9_]+?)(?P<sampled>_sampled_t[0-9]+(?:_cap[0-9]+)?)?$")
#: the padded lane's private runs (make_padding_configs.private_run): `pad_static_<key>[_<family>]_p<pad>`, family `obj` / `obj_s<N>`
#: (qwen35_4b's steered families) or a PHASE VIEW (`s2`, `d0p75_relu`, …: the front door's PER-VIEW steered families —
#: make_padding_configs.PHASE_VIEW_RE's grammar, restated so the key / family split of a run name stays unambiguous without the table)
_PADDED_RUN_RE = re.compile(r"^pad_static_(?P<key>[a-z0-9_]+?)(?:_(?P<fam>obj(?:_s[0-9]+)?|u0|s[0-9]+r?p?|s[0-9]+rl|dm?[0-9]+(?:p[0-9]+)?(?:_relul?)?))?_p(?P<pad>[0-9]+)$")   # PADDEDRELUL: the PHASE_VIEW_RE grammar exactly -- s2r / s2rl and _relul admitted


def _padded_key_family(private_run: str) -> Optional[Tuple[str, Optional[str]]]:
    """``(key, family)`` of a padded private run: the grammar split above, and — when the grammar names no family — the models table's
    DECLARED WAVE (demo-import M3: `pad_static_qwen35_4b_demo_battery_p0` -> (qwen35_4b_demo, battery)): the key group is split at a known
    front-door key (front_door_views.split_key_view, longest key first) and the split accepted iff the tail is one of that key's phase views
    (grammar or declared) — never a `battery` literal in the permissive regex. A key group that is itself a bare key (`pad_static_<key>_p0`,
    the prompted-only family) stays as it is. A missing table reader / table refuses by name (FrontDoorTableMissing), as for a dose tag."""
    m = _PADDED_RUN_RE.match(str(private_run or ""))
    if not m:
        return None
    key, fam = m.group("key"), m.group("fam")
    if fam is None:
        kv = FV.split_key_view(key)
        if kv and FV.is_phase_view(kv[0], kv[1]):
            return kv[0], kv[1]
    return key, fam


def family_of(row: Mapping[str, object], extra_families: Optional[Mapping[str, str]] = None) -> str:
    """The eval FAMILY of a pending row (the driver table's key): `--eval-family EVAL_ID=FAMILY` first, else the row's own `eval_category`
    through FAMILY_OF_CATEGORY; a static eval on a padded instrument is the padding family."""
    eval_id = str(row.get("eval_id") or "")
    if extra_families and eval_id in extra_families:
        fam = extra_families[eval_id]
    else:
        fam = FAMILY_OF_CATEGORY.get(str(row.get("eval_category") or ""), "unknown")
    return "padding" if fam == "static" and row.get("padding_tokens") is not None else fam


def store_root_of(family: str, store_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(root, private_run)`` a run's mapped store path names for its family's driver (FAMILY_LAYOUT): the root is ``depth`` components
    above the store, the private run the component right under the root (the config's stem for the lanes that have one). ``(None, None)``
    when the family has no layout or the path is too shallow to hold one -- the caller refuses the run by name."""
    layout = FAMILY_LAYOUT.get(family)
    if not layout or not store_path:
        return None, None
    depth = layout[0]
    parts = [p for p in os.path.abspath(str(store_path)).split(os.sep) if p]
    if len(parts) <= depth:
        return None, None
    root = os.sep + os.sep.join(parts[:-depth])
    return root, parts[-depth]


def target_for(family: str, private_run: Optional[str], model_key: str) -> Optional[str]:
    """The ``--target`` the family's grader holds its config to: the TAG of a sampled / multi-turn / BoN private run, the model key of a
    padded one (the graders assert `key == target` against the config's own spelling); None when the private run is not the family's."""
    kind = (FAMILY_LAYOUT.get(family) or (0, None))[1]
    if kind == "private_run":
        m = _FULL22_RUN_RE.match(str(private_run or ""))
        return m.group("tag") if m else None
    if kind == "padding":
        kf = _padded_key_family(str(private_run or ""))
        return kf[0] if kf else None
    return model_key or None


def config_for(family: str, *, configs_root: str, private_run: Optional[str] = None, model_key: Optional[str] = None) -> Optional[str]:
    """The EXPLICIT config a family's driver needs, under ``configs_root`` (the generators' own
    spellings): sampled / multi-turn / BoN → ``<configs_root>/<private_run>.yaml`` (make_full22_configs / make_sampled_configs name a
    config after its ``private_run``; the graders assert ``cfg["private_run"] == private_run``); padded → ``<configs_root>/sdfbase/
    static_padding/sdfbase_full22_<key>.padstatic[_<family>]_p<pad>.yaml`` (make_padding_configs.padded_config for its
    ``private_run``); Petri → ``<configs_root>/capeval/manifests/petri_full22_<model_key>.yaml`` (f22_resolve_config's MANIFEST for
    the TAG). None for a family with no config, or a private run the family's naming does not recognise."""
    kind = (FAMILY_LAYOUT.get(family) or (0, None))[1]
    if kind == "private_run":
        return os.path.join(configs_root, f"{private_run}.yaml") if _FULL22_RUN_RE.match(str(private_run or "")) else None
    if kind == "padding":
        m = _PADDED_RUN_RE.match(str(private_run or ""))
        kf = _padded_key_family(str(private_run or ""))
        if not m or not kf:
            return None
        key, family = kf
        fam = f"_{family}" if family else ""
        return os.path.join(configs_root, "sdfbase", "static_padding", f"sdfbase_full22_{key}.padstatic{fam}_p{int(m.group('pad'))}.yaml")
    if kind == "petri_manifest":
        return os.path.join(configs_root, "capeval", "manifests", f"petri_full22_{model_key}.yaml") if model_key else None
    return None


def _kv(specs: Optional[Sequence[str]], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in specs or ():
        if "=" not in s:
            raise SystemExit(f"{what} {s!r}: expected KEY=VALUE")
        k, _, v = s.partition("=")
        out[k.strip()] = v
    return out


_LOG_LOCK = threading.Lock()   # the grading loop and the Petri pass (its own thread) both write the event log


def log_event(state_dir: Optional[str], event: Mapping[str, object]) -> None:
    line = json.dumps(dict(event, utc=utc_now()), sort_keys=True, default=str)
    with _LOG_LOCK:
        print(f"[grade-daemon] {line}", flush=True)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
            with open(os.path.join(state_dir, "grade_daemon.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def pending_rows(root: str, model_ids: Sequence[str], campaign: Optional[str], store_root_map: Optional[P.StoreRootMap],
                 extra_families: Optional[Mapping[str, str]] = None, grader_session: Optional[str] = None) -> List[Dict[str, object]]:
    """The pending set for the models (each run's INSTRUMENT judge) through the shared worklist, so a row reaches the daemon located
    exactly as the drivers see it (`RegistryWorklist.pending`: the run's `substrate` and `view_legacy`, the instrument's
    `padding_tokens`, the mapped `store_path`), plus its `family` and the driver ROOT / private run its store names (`store_root_of`).
    A row ANOTHER session has claimed is left to that session -- the drivers' own rule: neither they nor the daemon take a claim over
    without `--force`. A row claimed by the daemon's OWN graders' session (`worklist.default_session()`: the session every grader it launches
    claims under) is dispatchable: the daemon runs its graders synchronously, so at tick time the grader that claimed it has exited -- a
    grader that died between its claim and its filing (2026-09-30: 16 runs left `judging` by graders that lost the write lock at their
    NEXT claim) would otherwise hold its own runs out of the worklist for `stale_hours`; the re-dispatched grader's claim returns the same
    judgment id (`pending.claim`: same session -> same claim). Such rows carry `reclaim: True`."""
    wl = RW.RegistryWorklist(root, None, "grade-daemon", campaign, driver=DRIVER_ID, model_ids=list(model_ids), store_root_map=store_root_map)
    own = grader_session or RW.default_session()
    out: List[Dict[str, object]] = []
    for row in wl.pending():
        held_by = row.get("claim_session")
        if held_by and str(held_by) != own:
            continue
        if held_by:
            row["reclaim"] = True
        row["family"] = family_of(row, extra_families)
        row["store_root"], row["private_run"] = store_root_of(str(row["family"]), row.get("store_path"))
        out.append(row)
    return out


def group_key(r: Mapping[str, object]) -> Tuple[str, str, str, str, str, str]:
    """One dispatch per (family, model key, campaign, driver root, private run[, eval id]): a driver invocation reads ONE root and ONE
    config (two sampled trials of a view are two configs), and the padded grader ONE (level, attack)."""
    family = str(r["family"])
    return (family, str(r.get("view_legacy") or r.get("model_id") or ""), str(r.get("campaign") or ""), str(r.get("store_root") or ""),
            str(r.get("private_run") or ""), str(r.get("eval_id") or "") if family in ONE_EVAL_PER_DISPATCH else "")


def group_rows(rows: Iterable[Mapping[str, object]]) -> Dict[Tuple[str, str, str, str, str, str], List[Dict[str, object]]]:
    groups: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, object]]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(dict(r))
    return groups


def placeholders(template: str) -> List[str]:
    """The `{name}` fields a template (or an --extra string) names, in order, once each."""
    out: List[str] = []
    for _lit, field, _spec, _conv in string.Formatter().parse(template):
        if field and field not in out:
            out.append(field)
    return out


def group_fields(rows: Sequence[Mapping[str, object]], *, python: str, repo: str, registry_root: str, store_root_map_specs: Sequence[str],
                 model_key: str, configs_root: str, grades_root: Optional[str], served_identity_dir: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Every placeholder value a group can supply, quoted for the shell; a derived value the group cannot supply is None (the caller
    refuses a template that names it). `store_root` / `private_run` are the group's ONE root / private run (the group key holds them
    equal across its rows); `config` is the family's explicit config under `configs_root`; `target` the grader's tag."""
    family = str(rows[0]["family"])
    store_root = rows[0].get("store_root")
    private_run = rows[0].get("private_run")
    target = target_for(family, private_run, model_key)
    config = config_for(family, configs_root=configs_root, private_run=private_run, model_key=model_key)
    attacks = ",".join(sorted({str(r["eval_id"]) for r in rows}))
    q = shlex.quote
    # the registry's model identity (the `model` table's hf_repo / hf_revision for the group's model key) for graders whose
    # rows step files model_id / model_revision -- the mirror carries the rows' store paths, not the judged root's identity files (the
    # 2026-10-02 agentic pass judged 2995 items and then failed its rows step on "no model identity"). Empty when the table does not
    # carry both. The grader uses these as a CROSS-CHECK against the identity its root's files state (mirrored
    # by mirror_root_identity) and refuses by name on a mismatch -- never as the identity it files.
    model_identity_flags = ""
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            hit = conn.execute("SELECT hf_repo, hf_revision FROM model WHERE model_id = ?", (str(rows[0].get("model_id") or ""),)).fetchone()
        if hit and hit[0] and hit[1]:
            model_identity_flags = f"--identity-model-id {q(str(hit[0]))} --identity-model-revision {q(str(hit[1]))}"
    except Exception:  # noqa: BLE001 -- no registry answer: no flags; the grader's own refusal names them
        model_identity_flags = ""
    return {
        "python": q(python), "repo": q(repo), "registry_root": q(registry_root),
        "run_id_flags": " ".join(f"--run-id {q(str(r['run_id']))}" for r in rows),
        "store_map_flags": " ".join(f"--store-root-map {q(s)}" for s in store_root_map_specs),
        "model_id": q(str(rows[0].get("model_id") or "")), "model_key": q(model_key), "campaign": q(str(rows[0].get("campaign") or "")),
        "attacks": attacks or None,
        "model_identity_flags": model_identity_flags,
        "target": q(target) if target else None, "config": q(config) if config else None,
        "store_root": q(str(store_root)) if store_root else None, "private_run": q(str(private_run)) if private_run else None,
        "grades_root": q(grades_root) if grades_root else None,
        # the dispatch's OWN served-identity snapshot directory (served_identity_dir_for; filled by snapshot_served_identity under
        # the root's identity lock before the grader starts) -- the grader's one read path for the served identity, never the shared root
        "served_identity_dir": q(served_identity_dir) if served_identity_dir else None,
    }


def build_command(template: str, rows: Sequence[Mapping[str, object]], *, python: str, repo: str, registry_root: str,
                  store_root_map_specs: Sequence[str], extra: str, model_key: str, configs_root: str = os.path.join(REPO, "configs"),
                  grades_root: Optional[str] = None, served_identity_dir: Optional[str] = None) -> List[str]:
    """The family's command line for one group; raises :class:`DispatchRefused` when the template (or `--extra`) names a placeholder
    the group cannot supply, or names a `{config}` that does not exist on this host."""
    fields = group_fields(rows, python=python, repo=repo, registry_root=registry_root, store_root_map_specs=store_root_map_specs,
                          model_key=model_key, configs_root=configs_root, grades_root=grades_root, served_identity_dir=served_identity_dir)
    needed = [f for f in placeholders(template) + placeholders(extra) if f != "extra"]
    unknown = [f for f in needed if f not in fields]
    if unknown:
        raise DispatchRefused("unknown_placeholder", f"the template names placeholders this daemon does not define: {unknown}")
    missing = [f for f in needed if fields[f] is None]
    if missing:
        why = {"store_root": "no store path on this host (store_uri unmapped or too shallow for the family's layout)",
               "private_run": "the store path holds no private run under the driver's root",
               "target": "the private run is not one the family's grader names a target for",
               "config": "the private run / model key resolve to no config (config_for)",
               "grades_root": "pass --grades-root (the graders' output root)",
               "served_identity_dir": "the family's root carries no served-identity files to snapshot (ROOT_IDENTITY_FILES), or the tick derived no dispatch directory",
               "attacks": "the group names no eval id"}
        raise DispatchRefused("placeholder_unresolved", "; ".join(f"{{{f}}}: {why.get(f, 'unresolved')}" for f in missing))
    if "config" in needed:
        cfg = shlex.split(str(fields["config"]))[0]
        if not os.path.isfile(cfg):
            raise DispatchRefused("config_missing", f"the driver's config {cfg} does not exist (generated by 05_stage_roster.sh --step configs "
                                                    f"for a front-door model of reproduce/models.yaml; committed for the paper's)", path=cfg)
    # the family's own arguments may use the same placeholders (`--target {model_key}`): substituted before they join the template
    cmd = template.format(**fields, extra=extra.format(**fields, extra=""))
    return shlex.split(cmd)


class DispatchRefused(Exception):
    """A group the daemon will not dispatch: `reason` is the event's short code, `detail` names what was missing."""

    def __init__(self, reason: str, detail: str, path: Optional[str] = None) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.path = path          # The config the refusal names, when it names one (the event carries it as `config_path`)


#: The runs REFUSED so far, run id -> (reason, detail, path), per process (the live daemon refused all 24 s1 static runs every tick for ~an hour
#: -- `config_missing`, the tree lacked the view's configs -- and the only trace was the tick line's `refused: 24` counter, which no watcher read).
#: `dispatch_refused` is said once per group whose runs carry a NEW (reason, detail) -- a run already refused for the same reason says nothing
#: again, a changed reason is said again -- naming the run ids, the config path and the daemon sha; `dispatch_refusal_cleared` once per run when
#: it dispatches or leaves the pending set; and every tick with a refusal raises a `problem` (kind dispatch_refused) beside the counter.
_REFUSED_RUNS: Dict[str, Tuple[str, str, Optional[str]]] = {}


def refusal_clear(state_dir: Optional[str], run_ids: Iterable[str], why: str, family: Optional[str] = None) -> List[str]:
    """Forget the refusal of every run in `run_ids` that was refused, one `dispatch_refusal_cleared` each; returns the cleared ids."""
    out: List[str] = []
    for rid in run_ids:
        was = _REFUSED_RUNS.pop(str(rid), None)
        if was is None:
            continue
        out.append(str(rid))
        log_event(state_dir, {"event": "dispatch_refusal_cleared", "run_id": str(rid), "family": family, "why": why, "was_reason": was[0], "was_detail": was[1],
                              "was_config_path": was[2], "note": "the run was refused by the dispatch before (dispatch_refused) and now " + why})
    return out


#: THE GRADE MIRROR'S HYGIENE: the mirror is a scoped cache under CLUSTER_GRADE_MIRROR_ROOT, never a copy of
#: the wave root. GUARD — a group's stores are mirrored only while every involved mirror root has at least this much free space; below it the
#: group's dispatch is SKIPPED for the tick (`mirror_skipped_low_disk`: grading paused, the wave untouched, retried next tick). PRUNE — the
#: store directories a tick rsynced for a group are queued in `<state-dir>/mirror_prune_pending.json` after the dispatch returns and removed
#: (`mirror_pruned`, with bytes) only once NO run of the group is still in `v_pending_judgment` — the worklist's OWN predicate
#: (`pending.pending(conn, None, campaign, run_id=…)`: the instrument's required judge with a complete judgment covering the required items),
#: never lifecycle or a judgment's status alone (a coverage-short or wrong-judge complete judgment keeps a run pending, and the worklist
#: would ask for its store again) — the harvester folds the drivers' spool with lag, so the confirmation is read on a later tick, never
#: assumed from the rc. Never the mirror root itself. The guard is BOUNDED by the stores' remote size when knowable (`rsync --dry-run
#: --stats`): a group larger than free − floor is skipped whole, never half-copied; and a tick whose rsync or dispatch failed RETIRES the
#: incomplete copies of its still-pending runs (`mirror_retired_incomplete`) before the next tick re-evaluates the floor, so a wedge (a
#: partial copy under the floor that nothing may re-mirror) cannot form. Two instances over ONE mirror (the daemon and a `--once`, or a
#: replacement daemon during a restart overlap) exclude each other per store directory: an exclusive flock on `<dir>.lock` beside each
#: mirrored store is held from before the copy through the dispatch and the settlement, and taken (non-blocking) by the prune pass before it
#: removes anything — an instance that cannot take a group's locks skips the group for the tick (`mirror_skipped_locked`) rather than copying
#: over, or retiring, a directory another instance is grading from. The GRADER inherits those lock fds (`pass_fds`), so the lock lives as long
#: as the reader: a daemon that dies under a running grader (a parent-only kill, an OOM of the parent) leaves the locks with the grader, and a
#: replacement skips the group until the grader is gone. Every dispatch is also a durable CLAIM (`mirror_dispatch_claims.json`: the daemon's
#: pid + start time, the group's runs and directories) that any later instance reconciles: a dead daemon's claim whose locks are free is
#: settled as an unknown outcome (still-pending copies retired, filed ones queued for the prune), so the space is recovered even when the
#: dispatching daemon never came back. The tick summary carries each root's du and free bytes.
MIRROR_MIN_FREE_BYTES_DEFAULT = 20 * 1024 ** 3
MIRROR_PRUNE_LEDGER = "mirror_prune_pending.json"
RSYNC_STATS_SIZE_RE = re.compile(r"Total transferred file size:\s*([\d,]+)\s*bytes")
#: TEST seam, armed ONLY under pytest: a stated free-space reading (bytes) that every mirror root reports in place of the real disk's. The
#: suite's daemon SUBPROCESSES set it (a daemon under test is a child of a running pytest, out of reach of an in-process monkeypatch): a
#: container filesystem can report an absurd free figure (~9.2e18 bytes, equal to its total, so ``db.free_bytes``' clamp to the total does
#: not help), a figure no floor a test can name exceeds, so the low-disk guard could never be exercised there. The variable is honoured only
#: while ``PYTEST_CURRENT_TEST`` is in the daemon's environment too (pytest sets it for each test's duration; a subprocess inherits it);
#: anywhere else it is a STRAY and the real reading stands -- a production daemon is never talked past the reserved floor by a leftover
#: variable, the low-disk preflight copies nothing onto a disk that is actually low, and the tick summary never reports fictitious capacity.
#: Neither state is silent: an honoured override is carried in every tick summary (``mirror_free_bytes_override``, the stated figure), an
#: ignored one in the summary as well (``mirror_free_bytes_override_ignored``, the raw value) and in one ``mirror_free_bytes_override_ignored``
#: event per tick, so a stray is named for as long as it stands rather than wondered about.
MIRROR_FREE_BYTES_ENV = "GRADE_DAEMON_FREE_BYTES"
PYTEST_CURRENT_TEST_ENV = "PYTEST_CURRENT_TEST"


def mirror_free_bytes_override() -> Optional[int]:
    """The figure ``GRADE_DAEMON_FREE_BYTES`` states while the test seam is ARMED (the variable set AND ``PYTEST_CURRENT_TEST`` non-empty in
    the environment), else None: unset, or set outside pytest (a stray -- the caller reads the real disk; ``mirror_free_bytes_override_ignored``
    names it). An armed value that is not a non-negative integer is refused by name rather than read as the real disk: a mis-set override
    must never pass as a healthy reading."""
    raw = os.environ.get(MIRROR_FREE_BYTES_ENV)
    if raw is None or not os.environ.get(PYTEST_CURRENT_TEST_ENV):
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{MIRROR_FREE_BYTES_ENV}={raw!r}: not an integer number of bytes") from None
    if value < 0:
        raise ValueError(f"{MIRROR_FREE_BYTES_ENV}={raw!r}: a free-space reading is never negative")
    return value


def mirror_free_bytes_override_ignored() -> Optional[str]:
    """The raw ``GRADE_DAEMON_FREE_BYTES`` the daemon is IGNORING -- set while ``PYTEST_CURRENT_TEST`` is not, i.e. outside pytest -- else None.
    Not parsed: a stray is named, never read, whatever it holds."""
    raw = os.environ.get(MIRROR_FREE_BYTES_ENV)
    return raw if raw is not None and not os.environ.get(PYTEST_CURRENT_TEST_ENV) else None


def mirror_free_bytes(root: str) -> int:
    """The daemon's one free-space reading for a mirror root: ``db.free_bytes`` (the filesystem's free bytes, clamped to its total) -- or,
    under pytest only, the figure the test seam states (``mirror_free_bytes_override``). Outside pytest the real reading is authoritative
    whatever the environment says."""
    override = mirror_free_bytes_override()
    return DB.free_bytes(root) if override is None else override


def mirror_roots_of(specs: Optional[Sequence[str]]) -> Dict[str, str]:
    """substrate -> this host's mirror root, from `--store-root-map SUBSTRATE=REMOTE_ROOT=LOCAL_ROOT` (the local roots ARE the mirror roots)."""
    out: Dict[str, str] = {}
    for spec in specs or ():
        parts = str(spec).split("=", 2)
        if len(parts) == 3 and parts[0].strip() and parts[2].strip():
            out[parts[0].strip()] = os.path.expanduser(parts[2].strip()).rstrip("/")
    return out


def _du(path: str) -> int:
    """Bytes of the regular files under `path` (no symlink following); 0 for an absent path."""
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for fn in files:
            try:
                st = os.lstat(os.path.join(dirpath, fn))
            except OSError:
                continue
            if not os.path.islink(os.path.join(dirpath, fn)):
                total += st.st_size
    return total


def _under_root(root: str, path: str) -> bool:
    """`path` is strictly below `root` (never the root itself, never outside it)."""
    r, p = os.path.realpath(root), os.path.realpath(path)
    return p != r and os.path.commonpath([r, p]) == r


def mirror_targets(rows: Sequence[Mapping[str, object]], mirror_ssh: Mapping[str, str]) -> List[Tuple[Mapping[str, object], str, str, str]]:
    """The rows this host mirrors — (row, ssh alias, remote store path, local store path): a substrate with an alias and a path the store map
    relocated (a path the map left unchanged is read where it is)."""
    out = []
    for r in rows:
        alias = mirror_ssh.get(str(r.get("substrate") or ""))
        remote = P.store_path_of(r)
        local = str(r.get("store_path") or "")
        if alias and remote and local and local != remote:
            out.append((r, alias, remote, local.rstrip("/")))
    return out


class MirrorLockHold:
    """What `mirror_locks` yields: the directories another instance holds (`busy`; empty = every lock taken) and the fds of the locks taken
    (`fds`; the dispatch passes them to the grader so the lock outlives a dying daemon for as long as the grader reads)."""

    def __init__(self, busy: List[str], fds: List[int]):
        self.busy = busy; self.fds = fds


@contextlib.contextmanager
def mirror_locks(dirs: Sequence[str]):
    """Exclusive per-store-directory locks: `<dir>.lock` BESIDE each mirrored store (never inside it — a removal must not take the lock's
    inode with it), flock non-blocking, in sorted order; all or none. Released by CLOSING the fds, never LOCK_UN (an unlock would also release
    the grader's inherited copy). Held by the tick from before the copy through the dispatch and the settlement — and by the grader itself,
    which inherits the fds — and by the prune pass around its removal, so no instance copies over, reads, or retires a directory another
    instance (or an orphaned grader) is working from."""
    fds: List[int] = []; busy: List[str] = []
    try:
        for d in sorted(set(dirs)):
            os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
            fd = os.open(d + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd); busy.append(d); continue
            fds.append(fd)
        if busy:
            for fd in fds:
                os.close(fd)
            fds = []
        yield MirrorLockHold(busy, list(fds))
    finally:
        for fd in fds:
            os.close(fd)


MIRROR_DISPATCH_CLAIMS = "mirror_dispatch_claims.json"


def _claims_load(state_dir: Optional[str]) -> List[dict]:
    if not state_dir:
        return []
    try:
        with open(os.path.join(state_dir, MIRROR_DISPATCH_CLAIMS), encoding="utf-8") as fh:
            doc = json.load(fh)
        return list(doc.get("claims") or []) if isinstance(doc, dict) else []
    except (OSError, ValueError):
        return []


def _claims_save(state_dir: str, claims: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(state_dir, exist_ok=True)
    tmp = os.path.join(state_dir, MIRROR_DISPATCH_CLAIMS) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"claims": list(claims), "utc": utc_now()}, fh, indent=2, sort_keys=True); fh.write("\n")
    os.replace(tmp, os.path.join(state_dir, MIRROR_DISPATCH_CLAIMS))


def mirror_claim_dispatch(state_dir: Optional[str], *, family: str, model_key: str, campaign: Optional[str], run_ids: Sequence[str],
                          mirrored: Sequence[Mapping[str, object]]) -> Optional[str]:
    """Before a group's dispatch: a DURABLE claim naming this daemon (pid + /proc start time), the group's runs and the directories it
    mirrored — so an instance that outlives this daemon can settle what it never came back to. Returns the claim id."""
    if not state_dir or not mirrored:
        return None
    pid = os.getpid()
    claim = {"id": f"{utc_now()}-{pid}-{len(_claims_load(state_dir))}", "daemon_pid": pid, "daemon_start": _proc_start(pid), "family": family,
             "model_key": model_key, "campaign": campaign, "run_ids": sorted(set(run_ids)), "mirrored": list(mirrored), "claimed_utc": utc_now()}
    claims = _claims_load(state_dir); claims.append(claim); _claims_save(state_dir, claims)
    return str(claim["id"])


def mirror_release_dispatch(state_dir: Optional[str], claim_id: Optional[str]) -> None:
    if not state_dir or not claim_id:
        return
    _claims_save(state_dir, [c for c in _claims_load(state_dir) if c.get("id") != claim_id])


def mirror_reconcile_claims(state_dir: Optional[str], registry_root: str, campaign: Optional[str]) -> Dict[str, int]:
    """At the start of a tick: a claim whose daemon is DEAD (pid + start time gone) names a dispatch nobody will settle. Its grader may still
    be running — the store locks it inherited say so: busy → the claim is kept (`mirror_claim_deferred`); free → the group is settled as an
    unknown outcome (`mirror_settle(rc=None)`: still-pending copies retired, filed ones queued for the prune) and the claim dropped
    (`mirror_claim_reconciled`). A LIVE daemon's claim is that instance's business (its locks exclude this one). Returns counts."""
    out = {"reconciled": 0, "deferred": 0}
    claims = _claims_load(state_dir)
    if not claims:
        return out
    keep: List[dict] = []
    for c in claims:
        pid, start = c.get("daemon_pid"), c.get("daemon_start")
        if isinstance(pid, int) and _proc_alive(pid, start if isinstance(start, int) else None):
            keep.append(c); continue
        dirs = [str(m.get("local") or "") for m in c.get("mirrored") or [] if m.get("local")]
        with mirror_locks(dirs) as held:
            if held.busy:
                out["deferred"] += 1
                log_event(state_dir, {"event": "mirror_claim_deferred", "claim": c.get("id"), "daemon_pid": pid, "dirs": held.busy, "run_ids": c.get("run_ids"),
                                      "note": "the dispatching daemon is gone but its grader still holds these directories: left alone until it exits"})
                keep.append(c); continue
            st = mirror_settle(state_dir, registry_root, family=str(c.get("family") or ""), model_key=str(c.get("model_key") or ""), campaign=c.get("campaign") or campaign,
                               run_ids=[str(x) for x in c.get("run_ids") or []], mirrored=list(c.get("mirrored") or []), rc=None,
                               reason="the daemon that dispatched this group died before settling it (its grader has exited)")
            out["reconciled"] += 1
            log_event(state_dir, {"event": "mirror_claim_reconciled", "claim": c.get("id"), "daemon_pid": pid, "run_ids": c.get("run_ids"), **st})
    _claims_save(state_dir, keep)  # type: ignore[arg-type]
    return out


def mirror_store_size(rsync_bin: str, src: str, dst: str) -> Optional[int]:
    """The bytes an rsync of `src` into `dst` would transfer (`rsync --dry-run --stats`, over the same ssh path), or None when unknowable — a
    failed probe never blocks a mirror, it only leaves the guard unbounded for that store."""
    try:
        r = subprocess.run([rsync_bin, *RSYNC_FLAGS, "--dry-run", "--stats", src, dst], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    m = RSYNC_STATS_SIZE_RE.search(r.stdout or "")
    return int(m.group(1).replace(",", "")) if m else None


def mirror_stores(rows: Sequence[Mapping[str, object]], mirror_ssh: Mapping[str, str], rsync_bin: str, dry_run: bool, state_dir: Optional[str],
                  mirror_roots: Optional[Mapping[str, str]] = None, min_free_bytes: int = MIRROR_MIN_FREE_BYTES_DEFAULT,
                  identity_snapshot_dir: Optional[str] = None) -> Optional[List[Dict[str, object]]]:
    """rsync each row's store from its cluster (`--mirror <substrate>=<ssh alias>`) to the mapped local path (the recorded path relocated
    through the store map). A row whose substrate has no alias, or whose path the map left unchanged, is not mirrored (the driver reads it
    where it is). GUARD: before the first rsync every involved mirror root must keep `min_free_bytes` free AFTER the group's stores land
    — their size probed with `rsync --dry-run --stats` where knowable (an unknowable size leaves the floor check plain) — else NOTHING is
    mirrored and None is returned: the caller skips the group's dispatch for this tick (a group larger than the headroom is skipped whole,
    never half-copied). Otherwise returns the store directories this call rsynced ({run_id, substrate, local, root, rc, bytes}; a failed
    rsync is logged and its directory still listed, so the caller's settlement retires or queues it) — never fatal: the driver refuses a
    run by name if it cannot locate the store. ``identity_snapshot_dir``: the dispatch's own served-identity snapshot
    directory, filled right after each judged root's identity transfer in the same locked section (mirror_root_identity)."""
    todo = mirror_targets(rows, mirror_ssh)
    roots = {str(r.get("substrate") or ""): (mirror_roots or {}).get(str(r.get("substrate") or "")) for r, _a, _re, _l in todo}
    ident_roots: Dict[Tuple[str, str, str], str] = {}       # (alias, remote root, local root) -> family, mirrored once after the stores
    if not dry_run and todo:
        need: Dict[str, int] = {}; unknown: Dict[str, int] = {}
        for r, alias, remote, local in todo:
            sub = str(r.get("substrate") or "")
            os.makedirs(local, exist_ok=True)
            size = mirror_store_size(rsync_bin, f"{alias}:{remote.rstrip('/')}/", local.rstrip("/") + "/")
            if size is None:
                unknown[sub] = unknown.get(sub, 0) + 1
            else:
                need[sub] = need.get(sub, 0) + size
        for sub, root in sorted(roots.items()):
            if not root:
                continue
            os.makedirs(root, exist_ok=True)
            free = mirror_free_bytes(root)
            if free - need.get(sub, 0) < min_free_bytes:
                log_event(state_dir, {"event": "mirror_skipped_low_disk", "substrate": sub, "mirror_root": root, "free_bytes": free, "need_bytes": min_free_bytes,
                                      "store_bytes": need.get(sub, 0), "unknown_sizes": unknown.get(sub, 0), "run_ids": [str(r["run_id"]) for r, _a, _re, _l in todo],
                                      "note": "grading paused for this group; the wave is untouched; nothing copied; retried next tick"})
                return None
    mirrored: List[Dict[str, object]] = []
    for r, alias, remote, local in todo:
        cmd = [rsync_bin, *RSYNC_FLAGS, f"{alias}:{remote.rstrip('/')}/", local.rstrip("/") + "/"]
        log_event(state_dir, {"event": "mirror", "run_id": r["run_id"], "cmd": " ".join(shlex.quote(c) for c in cmd), "dry_run": dry_run})
        if dry_run:
            continue
        os.makedirs(local, exist_ok=True)
        try:
            rc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200).returncode
        except (OSError, subprocess.SubprocessError) as exc:
            rc = f"error: {exc}"
        if rc != 0:
            log_event(state_dir, {"event": "mirror_failed", "run_id": r["run_id"], "rc": rc})
        mirrored.append({"run_id": str(r["run_id"]), "substrate": str(r.get("substrate") or ""), "local": local.rstrip("/"),
                         "root": roots.get(str(r.get("substrate") or "")), "rc": rc, "bytes": _du(local)})
        fam = str(r.get("family") or "")
        if fam in ROOT_IDENTITY_FILES:
            remote_root, _pr = store_root_of(fam, remote)
            local_root, _pl = store_root_of(fam, local)
            if remote_root and local_root:
                key = (alias, remote_root.rstrip("/"), local_root.rstrip("/"))
                ident_roots[key] = fam
                mirrored[-1]["identity_root"] = key[2]
    ident_rc: Dict[str, Union[int, str]] = {}
    for (alias, remote_root, local_root), fam in sorted(ident_roots.items()):
        ident_rc[local_root] = mirror_root_identity(alias, remote_root, local_root, ROOT_IDENTITY_FILES[fam], rsync_bin, state_dir, fam,
                                                    snapshot_dir=identity_snapshot_dir)
    for m in mirrored:                                   # the identity rc rides every entry of its root; the tick blocks on it
        if m.get("identity_root") in ident_rc:
            m["identity_rc"] = ident_rc[str(m["identity_root"])]
    return mirrored


def mirror_root_identity(alias: str, remote_root: str, local_root: str, files: Sequence[str], rsync_bin: str, state_dir: Optional[str],
                         family: str, snapshot_dir: Optional[str] = None) -> Union[int, str]:
    """ONE rsync of the judged root's identity files (``files``, relative to the root; a missing one is skipped --
    `--ignore-missing-args`) into the mirror's root, so the grader's rows step reads the SERVED identity as on the cluster. The
    enumerated LOCAL candidates are deleted BEFORE the transfer, so a file the current source no longer carries
    (a root reused by another served revision; a formerly preferred cells_full.json) can never persist from an earlier mirror and outrank
    the current identity in rows.resolve_model_identity; and the rc is the caller's to act on -- a nonzero rc blocks the group's dispatch
    for the tick (mirror_stores stamps it on the group's entries; the tick skips the group), never a dispatch on a partial identity set.
    The mirror root is SHARED by every group under it, and the per-store locks cover the stores,
    not the root -- so this delete-and-transfer by a second instance (a `--once`, a replacement daemon, selecting another newly generated
    cell under the same root) used to happen underneath a running grader that read the root at its rows step. Now the delete, the
    transfer and -- when the caller names the dispatch's ``snapshot_dir`` -- the per-dispatch SNAPSHOT of the files just transferred
    (snapshot_served_identity) run in ONE section under the root's identity lock (root_identity_lock, non-blocking: a root another
    instance is transferring right now is reported `locked`, nothing deleted, the group retried next tick), so the set a grader is handed
    cannot interleave with another instance's; a snapshot that fails is a nonzero rc like a failed transfer. The grader reads only its
    snapshot (`--served-identity-dir`): what happens to the root afterwards cannot reach it. Logged as `mirror_identity` with the files
    present after the transfer and the snapshot's files."""
    os.makedirs(local_root, exist_ok=True)
    removed: List[str] = []; snapped: Optional[List[str]] = None
    rc: Union[int, str]
    with root_identity_lock(local_root) as taken:
        if not taken:
            rc = "locked: another instance is transferring this root's identity files right now"
        else:
            for f in files:
                p = os.path.join(local_root, f)
                if os.path.isfile(p):
                    try:
                        os.remove(p); removed.append(f)
                    except OSError:
                        pass
            cmd = [rsync_bin, "-lt", "--no-owner", "--no-group", "--no-perms", "--ignore-missing-args", "--files-from=-", f"{alias}:{remote_root}/", local_root + "/"]
            try:
                rc = subprocess.run(cmd, input="\n".join(files) + "\n", capture_output=True, text=True, timeout=600).returncode
            except (OSError, subprocess.SubprocessError) as exc:
                rc = f"error: {exc}"
            if rc == 0 and snapshot_dir is not None:
                snapped = snapshot_served_identity(local_root, snapshot_dir, files, state_dir, family)
                if snapped is None:
                    rc = f"error: the dispatch's identity snapshot into {snapshot_dir} failed"
    log_event(state_dir, {"event": "mirror_identity", "family": family, "remote_root": remote_root, "local_root": local_root, "files": list(files), "rc": rc,
                          "removed_stale": removed, "present": [f for f in files if os.path.isfile(os.path.join(local_root, f))],
                          "snapshot": snapshot_dir, "snapshot_files": snapped})
    return rc


def identity_mirror_blocked(mirrored: Sequence[Mapping[str, object]]) -> List[str]:
    """The judged roots whose identity transfer failed this tick (``identity_rc`` not 0 on the group's mirrored entries): a non-empty list
    blocks the group's dispatch -- a grader must never run on a partial identity set."""
    return sorted({str(m.get("identity_root")) for m in mirrored if m.get("identity_rc") not in (None, 0)})


@contextlib.contextmanager
def root_identity_lock(local_root: str):
    """An exclusive, NON-BLOCKING flock on `<root>.identity.lock` beside the judged root (never inside it), yielding whether it
    was taken. Held across the identity transfer (delete, rsync) and the dispatch's snapshot copy -- seconds -- so two instances' identity
    operations on one root never interleave (one could otherwise snapshot between the other's delete and transfer, or copy a half-replaced
    set). Never held across a dispatch: the per-store locks (mirror_locks) are the long-lived ones; a caller that does not get it treats
    the identity as not transferred this tick."""
    root = local_root.rstrip("/")
    os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
    fd = os.open(root + ".identity.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            taken = True
        except OSError:
            taken = False
        yield taken
    finally:
        os.close(fd)


def served_identity_dir_for(log_path: str) -> str:
    """The dispatch's served-identity snapshot directory -- BESIDE its dispatch log and named after it plus this daemon's pid
    (`<log>.<pid>.identity/`), so it is keyed to this one dispatch: created once by snapshot_served_identity (never over an existing
    directory), read by the grader it is passed to (`--served-identity-dir`), re-mirrored and pruned by nothing."""
    base = log_path[:-len(".log")] if log_path.endswith(".log") else log_path
    return f"{base}.{os.getpid()}.identity"


def snapshot_served_identity(local_root: str, snapshot_dir: str, files: Sequence[str], state_dir: Optional[str], family: str) -> Optional[List[str]]:
    """The grader's IMMUTABLE per-dispatch copy of the judged root's served-identity files -- the
    ``files`` present under ``local_root``, copied into ``snapshot_dir`` with their relative layout kept (so rows.resolve_model_identity
    reads the snapshot exactly as it reads a root). The caller holds the root's identity lock. The directory must not exist yet (it is
    this dispatch's alone; an existing one is another dispatch's and is never reused or overwritten). Returns the files copied (possibly
    none: the grader then refuses by name at its preflight, exactly as a root with no identity does -- never a stale or borrowed identity),
    or None when the copy failed (`identity_snapshot_failed`; the caller does not dispatch the group this tick)."""
    copied: List[str] = []
    try:
        os.makedirs(snapshot_dir, exist_ok=False)
        for f in files:
            src = os.path.join(local_root, f)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(snapshot_dir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(f)
    except OSError as exc:
        log_event(state_dir, {"event": "identity_snapshot_failed", "family": family, "root": local_root, "snapshot": snapshot_dir, "files": list(files),
                              "error": f"{type(exc).__name__}: {exc}",
                              "note": "the dispatch's own copy of the judged root's identity files could not be taken: the group is not dispatched this tick"})
        return None
    log_event(state_dir, {"event": "served_identity_snapshot", "family": family, "root": local_root, "snapshot": snapshot_dir, "files": copied,
                          "note": "the grader reads the served identity from this directory only (--served-identity-dir); a later re-mirror of the root cannot reach it"})
    return copied


def _prune_ledger_path(state_dir: str) -> str:
    return os.path.join(state_dir, MIRROR_PRUNE_LEDGER)


def _prune_ledger_load(state_dir: Optional[str]) -> List[dict]:
    if not state_dir:
        return []
    try:
        with open(_prune_ledger_path(state_dir), encoding="utf-8") as fh:
            doc = json.load(fh)
        return list(doc.get("pending") or []) if isinstance(doc, dict) else []
    except (OSError, ValueError):
        return []


def _prune_ledger_save(state_dir: str, pending: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(state_dir, exist_ok=True)
    tmp = _prune_ledger_path(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"pending": list(pending), "utc": utc_now()}, fh, indent=2, sort_keys=True); fh.write("\n")
    os.replace(tmp, _prune_ledger_path(state_dir))


def mirror_prune_enqueue(state_dir: Optional[str], *, family: str, model_key: str, campaign: Optional[str], run_ids: Sequence[str],
                         mirrored: Sequence[Mapping[str, object]], rc) -> None:
    """After a group's dispatch returned: remember the store directories this tick rsynced for it, to be removed once NO run of the group is
    pending any more. Directories not strictly under a known mirror root are never queued (never the root, never outside it)."""
    if not state_dir or not mirrored:
        return
    dirs = [{"local": m["local"], "root": m["root"], "run_id": m["run_id"]} for m in mirrored if m.get("root") and _under_root(str(m["root"]), str(m["local"]))]
    if not dirs:
        return
    pending = _prune_ledger_load(state_dir)
    locals_new = {d["local"] for d in dirs}
    # a group queued again for the same directories (a retryable tick, then the clean one) REPLACES its earlier entry: one entry per copy
    pending = [e for e in pending if not (sorted(e.get("run_ids") or []) == sorted(set(run_ids)) and {d.get("local") for d in e.get("dirs") or []} <= locals_new)]
    pending.append({"family": family, "model_key": model_key, "campaign": campaign, "run_ids": sorted(set(run_ids)), "dirs": dirs, "dispatch_rc": rc, "queued_utc": utc_now()})
    _prune_ledger_save(state_dir, pending)


# ------------------------------------------------------------------------------------------------ the dispatch quarantine (2026-09-30)
#: A dispatch that FAILS must not be repeated every tick. Before this, a grader that died at its claim (`LockHeld` on the registry's write
#: lock, held by a back-to-back harvester) made every tick re-mirror the same ~20 stores, launch the same graders, watch them die and retire
#: the copies (the 7e0b4a6a5 rule) — rsync bandwidth and launches burned with no progress and nothing raised. Now a run whose dispatch failed
#: is QUARANTINED: recorded in `<state-dir>/dispatch_quarantine.json` (`{run_id, family, rc, stderr_tail, daemon_sha, store_generation, cmd,
#: dispatch_log, utc}`, one `dispatch_failed` event per group carrying the same) and neither mirrored nor dispatched by a later tick until its
#: CAUSE changes: a new daemon code sha (`daemon_code_sha`: the git HEAD of this file's checkout, `+<file digest>` when the file differs from
#: it), a changed store generation (the run's `store_uri`, `n_items_generated`, `store_completion_sha256`, `generated_utc` — a re-closed or
#: regenerated store), or an explicit release file `<state-dir>/dispatch_quarantine.release` (one run id per line, or `*`; consumed when read).
#: A quarantined run that leaves the pending view (judged by another route, superseded) is dropped from the ledger. Every tick with a held
#: run raises ONE `problem` event (`kind: dispatch_quarantine`, the count and the run ids) and carries `quarantined` + `problems` in its
#: summary, so the hold is visible instead of silent. Retirement of the failed tick's copies stays: the quarantine adds only the hold.
#: The grader's output goes to `<state-dir>/dispatch_logs/<utc>_<family>_<model key>.log` (a temp dir without a state dir) so a failure's
#: tail can be recorded; the `dispatched` event names the file.
#: The SPOOLED hold: under the harvester-folds contract a grader that exits clean has its judgments in the spool,
#: not in the database; until the harvester folds them the run still reads pending (its own-session claim would even be RECLAIMED), and a
#: 120 s tick would re-dispatch it — repeated paid judging, repeated rsync, duplicate records. A clean dispatch therefore writes one durable
#: hold per run (`dispatch_spooled.json`), honoured by the selection until the run leaves its instrument judge's pending view (the fold
#: observed), then cleared (the prune pass takes the copy). THE HOLD IS BOUND TO ITS DISPATCH: it names the complete
#: judgment record(s) THIS dispatch's grader spooled (`judgment_ids`, read off the run's spool directory after the grader exits, written at or
#: after the dispatch started) and the store generation it graded (`store_generation`, as the quarantine stamps it) — never "any historical
#: complete judgment in the directory", which held a run whose grader filed 198 of 200 items, or whose store was regenerated beside an old
#: complete file, for ever. Each tick, per hold: the run left the pending view → cleared (fold observed); the store's generation changed →
#: cleared `store_regenerated` (the graded store is gone; the run is dispatchable); a bound judgment is FOLDED yet the run still pends
#: (`coverage_short`, a wrong-judge complete) → the hold is released and the run QUARANTINED under a NAMED coverage defect (the judgment id,
#: its items against the required coverage; the quarantine's release paths are the controlled retry — a new daemon sha, a re-filed store,
#: or the release file), never re-dispatched blindly; a bound judgment no longer in the spool and not folded → dropped (nothing to wait for:
#: reclaimable); `dispatch_spooled.release` naming the run (or `*`) → dropped by the operator (an unresolved spool refusal). A clean grader
#: that filed NO complete judgment gets no hold (the dead-grader reclaim path stands). A hold older than SPOOLED_HOLD_WARN_S raises a
#: `problem` (spooled, never folded: the harvester or a refusal) instead of being released. A ledger entry without `judgment_ids` (written
#: by a daemon before this rule) keeps the old proof — a complete judgment in the directory — and is logged `legacy` until it clears.
DISPATCH_SPOOLED = "dispatch_spooled.json"
DISPATCH_SPOOLED_RELEASE = "dispatch_spooled.release"
SPOOLED_HOLD_WARN_S = 3600.0
DISPATCH_QUARANTINE = "dispatch_quarantine.json"
DISPATCH_QUARANTINE_RELEASE = "dispatch_quarantine.release"
#: The pending_reason v_pending_judgment gives a run -- on the SAME (run_id, required_judge_id) row -- that has no identity-current covering
#: judgment for its required judge AND a judgment (in_progress, partial or complete) for that judge whose stated binding
#: (claimed_instrument_id / claimed_required_judge_id) differs from the run's current instrument / required judge: the old judgment's row is
#: untouched (status unchanged, still in existing_judgments_json) but counts for nothing; a stale-bound in-progress claim reads this, never
#: `claimed` (pending.claim closes it and mints a current-bound one). ONE spelling, here (mirrored in scripts/petri_run.py), so a spelling
#: change is a one-token edit. Not produced by the view today.
IDENTITY_STALE_REASON = "identity_stale"
#: A run's IDENTITY BINDING is the pair the pending view states for it -- the run's `instrument_id` and its `required_judge_id` (the
#: instrument's judge). The spooled hold and the quarantine record it at filing time (the identity the grader was dispatched against), the
#: fold check compares it against the run's CURRENT binding (and against the binding a stale judgment itself states --
#: `claimed_instrument_id` / `claimed_required_judge_id` -- once the judgment table carries those columns; until then the daemon's own
#: binding at dispatch stands in), and the quarantine releases on a VERIFIED change of it for the same run: never a blanket release on the
#: reason alone.
#: The identity_stale branch once keyed the clear on the pending REASON, and the view gives that reason only on the required judge's own
#: row -- when the REQUIRED JUDGE itself moved J1 -> J2 under a dispatch, J1's folded completion is 'another judge's' on the J2 row
#: (`awaiting_second_judge`), so the fold fell to `_quarantine_folded_short`, which recorded the CORRECTED binding, after which the
#: quarantine's identity_rebound release could never fire and J2 was never dispatched without an operator release. The trigger is therefore
#: the BINDING CHANGE itself (`identity_binding_differs(bound, current)`), whatever the reason says, and the judge the folded judgments must
#: be by is the one REQUIRED AT DISPATCH -- the hold's bound judge, not the current `required_judge_id`
#: (`binding_changed_clear_verdict`: this run's judgments, by the bound judge, under a binding that differs from the current one; the same
#: three checks the identity_stale path runs). Verified: cleared `binding_changed_rebind` for re-dispatch under the current identity (event
#: `spooled_hold_binding_changed`, both judges and both bindings named, so the audit tells it from identity_stale); refused (a foreign-run or
#: wrong-judge completion, no binding evidence): the coverage_short path as before, the verdict named. An UNCHANGED binding never enters it.
IDENTITY_BINDING_FIELDS: Tuple[str, ...] = ("instrument_id", "required_judge_id")
JUDGMENT_CLAIMED_BINDING_COLUMNS = {"instrument_id": "claimed_instrument_id", "required_judge_id": "claimed_required_judge_id"}


def identity_binding_of(row: Optional[Mapping[str, object]]) -> Optional[Dict[str, Optional[str]]]:
    """The run's identity binding as a pending-view row states it: ``{instrument_id, required_judge_id}`` (a field the row lacks is None);
    None for no row."""
    if row is None:
        return None
    return {k: (str(row.get(k)) if row.get(k) is not None else None) for k in IDENTITY_BINDING_FIELDS}


def identity_binding_differs(a: Optional[Mapping[str, object]], b: Optional[Mapping[str, object]]) -> bool:
    """Two bindings differ when BOTH are known and any field differs; an unknown side never counts as a change (no blanket release)."""
    if not a or not b:
        return False
    return any((a.get(k) if a.get(k) is not None else None) != (b.get(k) if b.get(k) is not None else None) for k in IDENTITY_BINDING_FIELDS)


def judgment_rows_with_binding(conn, judgment_ids: Sequence[str], *, exclude_in_progress: bool = True) -> List[Dict[str, object]]:
    """The bound judgments' rows: ``judgment_id, run_id, judge_id, status, n_items_judged`` plus the judgment's STATED binding under the daemon's
    field names (``stated_binding`` = {instrument_id, required_judge_id} read from `claimed_instrument_id` / `claimed_required_judge_id` when the
    judgment table has those columns, else None: the caller falls back to the daemon's own binding). ``run_id`` is the run the registry FOLDED
    the judgment under: a hold binds every complete judgment id in the run's spool directory, and a grader can write another run's completion
    through this run's writer (emit.RecordWriter._with_run keeps an explicit foreign run_id; pending.spool_complete_judgments reads the directory
    without inspecting it), so the verdict must see which run each folded judgment belongs to."""
    ids = [str(j) for j in judgment_ids]
    if not ids:
        return []
    cols = {r[1] for r in conn.execute("PRAGMA table_info(judgment)").fetchall()}
    claimed = {k: c for k, c in JUDGMENT_CLAIMED_BINDING_COLUMNS.items() if c in cols}
    sel = ["judgment_id", "run_id", "judge_id", "status", "n_items_judged"] + list(claimed.values())
    qmarks = ",".join("?" for _ in ids)
    sql = f"SELECT {', '.join(sel)} FROM judgment WHERE judgment_id IN ({qmarks})" + (" AND status <> 'in_progress'" if exclude_in_progress else "")
    out: List[Dict[str, object]] = []
    for x in conn.execute(sql, ids).fetchall():
        d = dict(zip(sel, tuple(x)))
        stated = {k: (str(d.pop(c)) if d.get(c) is not None else d.pop(c)) for k, c in claimed.items()} if claimed else None
        d["stated_binding"] = stated if stated and any(v is not None for v in stated.values()) else None
        out.append(d)
    return out


def _bound_fold_clear_verdict(folded: Sequence[Mapping[str, object]], *, run_id: str, required_judge: str, current: Optional[Mapping[str, object]],
                              bound: Optional[Mapping[str, object]], explains: str, judge_desc: str, bound_desc: str) -> Tuple[bool, str]:
    """The three checks a hold whose bound judgment(s) FOLDED while the run still pends must pass before it is CLEARED for re-dispatch:
    every folded judgment (1) is THIS run's -- its ``run_id`` is the held run (a judgment the registry folded under another run, written
    through this run's writer / directory, is not this run's fold, whatever its judge and binding say: run, judge and binding all count);
    (2) is by ``required_judge`` -- the judge the fold is explained for (the run's current required judge on the identity_stale face, the
    judge required AT DISPATCH on the binding-change face); and (3) was made under a binding (the judgment's stated binding when the row
    carries one, else the daemon's binding at dispatch) that differs from the run's CURRENT binding. ``explains`` / ``judge_desc`` /
    ``bound_desc`` word the verdict for the face. A failed check is ``(False, why)`` naming both runs / both judges; the coverage_short
    path then decides."""
    if not required_judge or not current:
        return False, f"no required judge or no current identity binding on the pending row: {explains} cannot be verified"
    foreign = [j for j in folded if str(j.get("run_id") or "") != str(run_id)]
    if foreign:
        names = ", ".join(f"foreign-run judgment {j.get('judgment_id')} folded under run {j.get('run_id')} (judge {j.get('judge_id')})" for j in foreign)
        return False, (f"{names} -- bound in run {run_id}'s hold (written through its spool directory) but not this run's judgment; {explains} "
                       "does not explain this fold")
    wrong = [j for j in folded if str(j.get("judge_id") or "") != str(required_judge)]
    if wrong:
        names = ", ".join(f"{j.get('judgment_id')} (judge {j.get('judge_id')})" for j in wrong)
        return False, (f"folded judgment(s) by another judge than {judge_desc} {required_judge}: {names} -- a wrong-judge completion, "
                       f"not {bound_desc}; {explains} does not explain this fold")
    sources: List[str] = []
    for j in folded:
        stated = j.get("stated_binding")
        binding, source = (stated, "the judgment's stated binding") if stated else (bound, "the daemon's binding at dispatch")
        sources.append(source)
        if not binding:
            return False, (f"no binding evidence for judgment {j.get('judgment_id')} (the hold predates the binding and the judgment row states none): "
                           f"{explains} cannot be verified against it")
        if not identity_binding_differs(binding, current):
            return False, (f"judgment {j.get('judgment_id')} ({source}) was made under the run's CURRENT identity {dict(current)}: the identity did not "
                           f"change under it; {explains} is not this judgment's")
    return True, (f"every folded judgment is run {run_id}'s, {judge_desc} {required_judge}'s, and was made under a binding "
                  f"({', '.join(sorted(set(sources)))}) that differs from the run's current identity {dict(current)}")


def identity_stale_clear_verdict(folded: Sequence[Mapping[str, object]], *, run_id: str, required_judge: str, current: Optional[Mapping[str, object]],
                                 bound: Optional[Mapping[str, object]]) -> Tuple[bool, str]:
    """Whether a hold whose bound judgment(s) FOLDED while the run pends as IDENTITY_STALE_REASON may be CLEARED -- only when every folded
    judgment IS the stale-bound one for THIS run's required judge: its ``run_id`` is the held run, its ``judge_id`` is ``required_judge`` (the
    judge required at dispatch when the hold names one, else the run's current required judge), AND the binding it was made under differs from
    the run's CURRENT binding (:func:`_bound_fold_clear_verdict`). A foreign-run judgment, a folded completion by ANOTHER judge (the wrong-judge
    case `_quarantine_folded_short` already handles), a hold with no binding evidence, or a binding equal to the current identity is NOT
    cleared: ``(False, why)`` and the coverage_short path decides, the verdict naming both runs / both judges."""
    return _bound_fold_clear_verdict(folded, run_id=run_id, required_judge=required_judge, current=current, bound=bound,
                                     explains="the identity_stale re-pend", judge_desc="the run's required judge", bound_desc="the stale-bound judgment")


def binding_changed_clear_verdict(folded: Sequence[Mapping[str, object]], *, run_id: str, current: Optional[Mapping[str, object]],
                                  bound: Optional[Mapping[str, object]]) -> Tuple[bool, str]:
    """DAEMONQUARFIX: whether a hold whose bound judgment(s) FOLDED while the run still pends -- under a run identity that CHANGED since the
    dispatch (the hold's ``bound`` binding differs from the run's ``current`` one), whatever the pending reason says -- may be CLEARED for
    re-dispatch under the current identity. The bound binding must name the judge required at dispatch (a rebuilt or legacy hold names none:
    refused, the coverage_short path decides) and must differ from the current binding (an unchanged binding is never this verdict's: refused
    by name); then the same three checks as the identity_stale face, against the DISPATCH-time judge: every folded judgment is this run's, by
    that judge (a completion by the CURRENT required judge bound in the hold is a wrong-judge completion here: not the dispatch's judgment), and
    made under a binding (stated, else the hold's) that differs from the current one."""
    if not bound or not bound.get("required_judge_id"):
        return False, ("the hold carries no bound required judge (written before the identity binding, or rebuilt from the spool): the binding change "
                       "cannot be verified against it")
    if not current:
        return False, "no current identity binding on the pending row: the binding change cannot be verified"
    if not identity_binding_differs(bound, current):
        return False, f"the hold's bound binding {dict(bound)} is the run's current identity: no binding change to explain the fold"
    return _bound_fold_clear_verdict(folded, run_id=run_id, required_judge=str(bound["required_judge_id"]), current=current, bound=bound,
                                     explains="the binding change", judge_desc="the judge required at dispatch",
                                     bound_desc="the judgment this dispatch was bound to")
#: The operator's dispatch HOLD -- while this file exists in the state dir a tick does its bookkeeping (mirror-claim
#: reconciliation, the spooled and quarantine passes) but dispatches NO grader and emits NO contrast: every pending row is left untouched
#: and one `held` event names the file's first line as the reason. Cleared by removing the file. The static correction op re-registers
#: units under rung-keyed run ids whose stores already carry complete judgments; the hold keeps the daemon from re-judging them while the
#: re-filed judgments are written and folded (the worklist's superseded-sibling skip is the belt behind it).
DISPATCH_HOLD = "dispatch.hold"
#: The operator's hold covers the Petri thread too -- `petri_drive_pass` settles and probes under it but STARTS no
#: drive (one `petri_drive_held` per tag naming the file; `petri_start_hold`) -- and the SCHEMA hold stands beside it (`schema_gap` / `SchemaGate`,
#: below `dispatch_hold`): while the live registry's migration version or shape is not the code's, the tick and the Petri start do no work at
#: all. How many drift lines a `schema_hold` event carries verbatim (the count is always carried).
SCHEMA_HOLD_DRIFT_SHOWN = 5
#: The mark of the last contrast re-emission SWEEP -- `{daemon_sha, utc, counts}`; the sweep runs again only when the
#: daemon's code advanced (IMPORTED_CODE_SHA differs), the same trigger that releases a quarantine stamped by an older import.
CONTRASTS_SWEEP_MARK = "contrasts_sweep.json"
#: The operator's HOLD on that sweep -- `<state-dir>/contrasts_sweep_hold`; while the file exists the once-per-advance
#: sweep does not run (contrasts_sweep_for returns before any registry read and writes no mark), its first non-comment line is the reason, and it
#: is read EVERY tick (never cached) so a flip takes effect at the next tick -- a file flip, never a unit restart (the 2026-10-04 release needed
#: one because `--no-contrasts-sweep` lived only in argv and os.execv keeps sys.argv). The argv flag is the INITIAL state only, applied ONCE per
#: state dir: the first flagged start writes the hold file (when absent) and the persisted record `<state-dir>/contrasts_sweep_hold.seeded`
#: (`contrasts_sweep_hold_initial`); every later start or code-drift re-exec with the flag finds the record and never recreates a released hold.
#: Operationally: a HOLD is a file touch, a RELEASE an rm, the flag seeds once. The file is read ONCE per tick, before the contrasts block, and
#: that snapshot is what the sweep is handed; the tick event reports the snapshot and what the sweep did (`contrasts_sweep_hold.sweep`: ran /
#: held_by_file / skipped_once_per_advance / operator_held / schema_held / ...).
CONTRASTS_SWEEP_HOLD = "contrasts_sweep_hold"
CONTRASTS_SWEEP_HOLD_SEEDED = "contrasts_sweep_hold.seeded"
#: sweephold-fix-2 (the UPGRADE path): the ABSENCE of `.seeded` is not permission to seed. A daemon on the parent code
#: (f03e6ee76) created the hold file but no record; an operator who released it (rm) while that daemon ran would have been re-held on the first
#: flagged re-exec into fix-1 (both files absent read as a first-ever start). Now a flagged start with no record seeds ONLY when the state dir
#: carries NO evidence of a previous daemon on sweephold code. The EVIDENCE (`sweephold_evidence`, exactly): (1) the state dir's own event log
#: `grade_daemon.jsonl` -- its first and last SWEEPHOLD_EVIDENCE_SCAN_BYTES -- carrying a `daemon_started` with `contrasts_sweep_hold_seeded` or
#: `contrasts_sweep_hold_seeding` (every sweephold-era start says one), a `contrasts_sweep_held` event, or a `tick` carrying
#: `contrasts_sweep_hold`; (2) the mark `contrasts_sweep.json` carrying `sweephold_era` (every mark written since fix-2 does). The hold file's
#: presence alone is NOT evidence (an operator may have touched it on a fresh dir) -- with no history it is left as it is and the record written,
#: fix-1's behaviour. With evidence and no record NOTHING is written: `daemon_started.contrasts_sweep_hold_seeding` says
#: `skipped: ambiguous_legacy_state` with the evidence named, and the OPERATOR decides -- `touch <state-dir>/contrasts_sweep_hold` to hold, or write
#: `<state-dir>/contrasts_sweep_hold.seeded` (any JSON object, e.g. {"acknowledged_by": "<who>"}) to acknowledge the released state; either ends
#: the ambiguity for every later start.
SWEEPHOLD_EVIDENCE_SCAN_BYTES = 4 * 1024 * 1024
SWEEPHOLD_SEEDING_EVENT_KEYS = ("contrasts_sweep_hold_seeded", "contrasts_sweep_hold_seeding")
SWEEPHOLD_SKIP_AMBIGUOUS = "ambiguous_legacy_state"
#: A DRY RUN is never evidence. A `--dry-run --no-contrasts-sweep` audit writes no hold file and no record (the carve-out) but its start-up and
#: ticks are logged like any other, and an earlier version counted them -- so the first REAL flagged start after an audit read
#: `ambiguous_legacy_state`, seeded nothing, and the campaign sweep ran against the requested hold. Now every event a dry-run daemon writes
#: carries `dry_run: true` (daemon_started, tick, contrasts_sweep_held) and `sweephold_history` EXCLUDES them -- evidence is a REAL
#: predecessor's seeding / seeded start, a REAL held tick, or a mark (a dry run writes no mark) -- reporting how many dry-run events it skipped
#: (`dry_run_events_skipped`) for the operator's reading; an older dry-run start-up (no stamp yet, its seeding dict saying `skipped: dry_run`)
#: is excluded the same way.
#: Stamping NEW dry-run start-ups and ticks is not enough on its own: an older dry-run audit left a start-up whose seeding dict says
#: `skipped: dry_run` FOLLOWED by ticks with no stamp at all, and those ticks still counted as a real hold era. Now every era event is
#: classified from its RUN CONTEXT (`sweephold_classify`): (a) a tick whose `contrasts_sweep_hold.sweep` reads `held_by_flag_dry_run`, or any
#: event stamped `dry_run: true`, is a dry-run event; (b) an event inside a dry-run start-up's WINDOW -- after a `daemon_started` that is a dry
#: run (stamp or `skipped: dry_run`) and before the next `daemon_started` or a `daemon_exit` (an event carrying a pid belongs to the window only
#: when the pid is the start-up's) -- is a dry-run event; (c) an event in a REAL start-up's window is genuine evidence; (d) an OLD unflagged
#: event that can be placed in no start-up window (its start-up outside the scanned text, or after a `daemon_exit` with no start since) is
#: UNCLASSIFIED: reported under `unclassified` (count, first / last utc), never counted as evidence, and never silently seeded over -- a flagged
#: start with unclassified history and no evidence is `skipped: unclassifiable_legacy_history` with the operator decision, so the flag never
#: turns on from history nobody can read.
#: Defaulting every unstamped `daemon_started` that does not say `skipped: dry_run` to a REAL window is wrong too. Two predecessor shapes defeat
#: it: f03e6ee76 (`--dry-run --no-contrasts-sweep`: `contrasts_sweep_hold_seeded: null`, no seeding dict, unstamped ticks with no `sweep`
#: status) and 00cf8b96c (`--dry-run` without the flag: `seeding.skipped: no_flag`, the no-flag return preceding the dry-run check, unstamped
#: ticks). Now an event classifies itself only by an AFFIRMATIVE discriminator (`_sweephold_event_class`): dry-run by the `dry_run: true` stamp,
#: the `skipped: dry_run` marker or a tick's `held_by_flag_dry_run`; real by the `dry_run: false` stamp or a seeding that WROTE a file
#: (`contrasts_sweep_hold_seeded` set, `seeded_now` set, `seeding_recorded`). An old start with neither is UNCLASSIFIED -- never real by
#: default -- and so are the ticks in its window; unclassified starts are counted under `unclassified.starts`. While a flagged start's state dir
#: carries unclassified history and nothing is seeded, the tick HOLDS the once-per-advance sweep (`SWEEP_STATUS_HELD_FLAG_PENDING`,
#: `contrasts_sweep_flag_pending`) until the operator decides; a readable history (the upgrade path: a real predecessor seeded a file, the
#: operator released it) still lets it run.
SWEEPHOLD_SKIP_UNCLASSIFIED = "unclassifiable_legacy_history"
#: A pending hold must not be re-derived EVERY tick from `sweephold_history`, which reads only the first and last
#: SWEEPHOLD_EVIDENCE_SCAN_BYTES of the log -- so once enough stamped ticks pushed the unclassified events out of the tail window (on the live
#: dir: 214 events at ~5.16-5.45 MB of a 5.59 MB log, none in the head; ~4 MB more of ticks and the window passes them) `pending` derived False
#: with no operator decision ever made and the sweep ran on a flagged start; a re-exec re-derived the same non-durable state.
#: Now the FIRST derivation of "pending" on a state dir (the start-up seeding path or the tick) PERSISTS it:
#: `<state-dir>/contrasts_sweep_pending_decision` (JSON, tmp + os.replace: the unclassified counts and first / last utc as `sweephold_history`
#: reported them, the real evidence beside them, the deriving daemon's sha / pid / utc, and the two operator paths as text). While the record
#: exists a flagged, non-dry tick HOLDS the sweep (`held_by_flag_pending_decision`) whatever the bounded scan sees -- the scan is not even read
#: -- and a flagged start seeds nothing (the decision is still open). Clearing is the OPERATOR's alone: the `.seeded` record (the release path),
#: honoured by the start-up or the tick, removes the pending record (`sweep_pending_decision_cleared`) and the sweep runs once per advance; the
#: hold file (the hold path) SUPERSEDES it while present (`held_by_file`; the record stays, so an rm of the file without a `.seeded` record holds
#: pending again). Never cleared by eviction, a re-exec, a later clean-looking scan, or an unflagged start (the record binds the FLAG's intent:
#: an unflagged daemon is not held by it and does not remove it). A dry run neither writes nor removes it. Written once and never rewritten (the
#: first derivation, dated); a record that exists but cannot be read still holds (the hold fails toward holding).
CONTRASTS_SWEEP_PENDING_DECISION = "contrasts_sweep_pending_decision"
#: The record's publish is RACE-FREE. Writing it through ONE fixed `.tmp` path and an unconditional os.replace let two real flagged starts on
#: one state dir race -- the loser died at start-up with FileNotFoundError (its tmp renamed away by the winner) before `daemon_started`, or both
#: published (a double record, two "one-time" recorded events). Now the body goes to a UNIQUE tmp (tempfile.mkstemp in the state dir) and is
#: published EXCLUSIVELY with os.link, which fails EEXIST when a winner's record exists; the loser reads the winner's record back and proceeds
#: held on it -- one record, one `sweep_pending_decision_recorded` event in total, the loser saying nothing of its own (its callers report
#: `written_now: false` with the winner's record) -- and its tmp is removed either way. The `.seeded`-driven clearing already tolerates the other
#: process removing first (FileNotFoundError -> None, no event). How many times the publish retries when the record it collided with has vanished
#: by the read-back (the operator's `.seeded` honoured in between, or an rm) before giving up on persisting this derivation (nothing pending is
#: reported; the next tick derives again):
SWEEPHOLD_PENDING_PUBLISH_ATTEMPTS = 5
CONTRASTS_SWEEP_HOLD_ARGV_REASON = "argv --no-contrasts-sweep (initial state)"
DISPATCH_LOG_DIR = "dispatch_logs"
DISPATCH_TAIL_CHARS = 4000
#: A failure whose cause is CONTENTION, not the run, the store or the grader: the registry's write lock held by the harvester when the
#: grader claimed. Read from the grader's TERMINAL exception — the last non-blank line of its output, which Python's traceback contract makes
#: `<module>.<Exception>: <message>` — never from a substring anywhere in the tail (an unrelated EAGAIN, or an early LockHeld followed by a
#: later fatal error, must be a defect). Such a dispatch is retried on the next tick with its copy KEPT; every other non-zero exit quarantines
#: (the retryable/real split, 2026-09-30). A structured `dispatch_outcome.json` from the grader would be the next step; no driver writes one
#: today.
RETRYABLE_TERMINAL = ("sisteer.registry.db.LockHeld:",)
#: Terminal lines that name UPSTREAM shedding rather than a defect: the live judge left items unscored after its own bounded retries (the JB1
#: rule refuses to aggregate a partial grade) — the judge pool's `overloaded_error` / 429 of the moment, the same class as the lock. These
#: are retried on later ticks with the copy kept, but BOUNDED: after RETRYABLE_CAP consecutive retryable failures the group is quarantined
#: with that reason (a pool that never recovers must become visible, not a grader relaunched every tick forever). The count is PER RUN and
#: charged by what the dispatch DID (the registry owner's ruling of 2026-10-13 23:29; `dispatch_progress` / `retryable_count`): a run the
#: dispatch filed a complete judgment for resets to 0 -- a dispatch that files at least one complete judgment is progress, not a
#: consecutive failure (the BoN leg files one cell per dispatch and leaves the next cell's items UNSCORED: nine dispatches of pure
#: progress read as six consecutive failures of the cells never reached and quarantined them two cells short, 2026-10-14) -- a run it
#: attempted and left incomplete increments, a run it never reached is not charged; only a dispatch that left no judgment record for any
#: run of the group is a failure of the whole group, every run charged (the lock refused before the first scope opened).
RETRYABLE_TERMINAL_SUBSTRINGS = ("still UNSCORED after live grading (bounded retries)",)
RETRYABLE_CAP = 6
#: A CREDENTIAL rejection -- the API answered 401 `authentication_error` (`invalid x-api-key`: the legacy lowprio key after the org's move to
#: federation) or 403 `PermissionDeniedError` -- is neither a defect of the run nor upstream shedding: EVERY request of EVERY grader of the
#: family fails the same way until the credential is fixed, so a retry costs 6 x N doomed requests per group and a quarantine hides the one
#: cause under hundreds of per-run entries (the demo smoke: 39 dispatches, 0 rc 0; 14 x 6 x 777 static + 6 x 7,500 BoN requests;
#: dispatch_retries.json at 6-7 for 60 run ids). Read from the TAIL, not only the terminal line (the one deliberate exception to the
#: terminal-line rule): the controls judge raises JudgingError whose message carries the failure-types histogram, the agentic judge records a
#: 401 per item and then dies on its rows step, the static graders end on the generic JB1 line -- the 401 text is wherever the grader printed
#: it, and a 401 / 403 anywhere in a grader's output is never benign (unlike the EAGAIN that rule excludes). The kind is `credential`: the
#: FAMILY is held (`dispatch.hold.<family>` -- the operator's own file, lifted by removing it), ONE `problem` of kind `judge_auth` names the
#: org and the secrets mode, the copy is kept, nothing is quarantined and nothing counts toward RETRYABLE_CAP.
#: THE FEDERATED SHAPE: under `SISTEER_SECRETS_MODE=wif` the credential is the federation identity, which the API refuses at the token
#: exchange -- the SDK's `WorkloadIdentityError(status_code=401 | 403)` with a REDACTED body that is an OAuth error string or a non-JSON text
#: as often as the API's error object, so the drivers' typed `sisteer.vault.JudgeCredentialRejected` reads `... rejected the <pool> pool's
#: credential: WorkloadIdentityError (HTTP 401) ...` with NONE of the legacy words above, and such a grader was `failed` and quarantined. The
#: regex therefore also reads the typed rejection's own clause (`rejected the <pool> pool's credential`, whatever SDK class it names), the
#: `(HTTP 401` / `(HTTP 403` the typed message carries, and the SDK's own `Token exchange failed (HTTP 40x)` line (a grader that let it
#: through raw). The class name ALONE never classifies: `WorkloadIdentityError: Failed to reach token endpoint` and the exchange's 5xx are
#: transient, the retry loops' business.
CREDENTIAL_TAIL_RE = re.compile(r"authentication_error|invalid x-api-key|\bAuthenticationError\b|\bPermissionDeniedError\b|Error code: 40[13]\b"
                                r"|rejected the \S+ pool's credential|\(HTTP 40[13][,)]")
#: the org a grader's tail names (`org=lowprio` on the static / BoN status lines, `pool lowprio` on the agentic judge's done line, the
#: provenance's `'pool': 'lowprio'`, the typed rejection's `rejected the lowprio pool's credential`); the LAST mention wins; none -> the
#: vault's default org, said so
_CREDENTIAL_ORG_RE = re.compile(r"(?:\borg=|\bpool\s+|['\"]pool['\"]:\s*['\"]|\bthe (?=(?:lowprio|highprio|batch) pool's credential))"
                                r"(lowprio|highprio|batch)\b")
CREDENTIAL_DEFAULT_ORG = "lowprio"
#: The JB1 UNSCORED line is retryable only as upstream SHEDDING. One whose embedded judge-client counts say the failures were TERMINAL
#: (`'terminal': 7493` against `'rate_limited': 747` -- the BoN grader's line), or whose failure-types histogram names a 4xx other than 429,
#: is `failed` (quarantined, visible), not `retryable` (six more relaunches of the same doomed group).
_JUDGE_COUNT_RE = {k: re.compile(rf"['\"]{k}['\"]:\s*(\d+)") for k in ("terminal", "rate_limited")}
_FAILURE_TYPES_4XX_RE = re.compile(r"Failure types:.*Error code: 4(?!29\b)\d\d\b")
#: A grader whose registry claim refused the database -- `pending.claim` -> `db.open_or_create` raises `db.SchemaDrift` when the
#: live migration version or shape is not the code's (a migration in flight under a running daemon; a daemon on code the database has not
#: caught up to) -- is TRANSIENT and HELD: the runs stay pending for the next tick's schema preflight, the copy is kept, nothing is
#: quarantined and nothing counts toward RETRYABLE_CAP (2026-10-02 20:53-21:30Z: `SchemaDrift(have=21, want=22)` read as a permanent failure,
#: quarantined the pending runs, and the COMPLETED migration changed none of the quarantine's release markers). Read from the terminal line
#: by the exception's qualified name (Python's traceback contract, as RETRYABLE_TERMINAL), never from the message's prose;
#: `schema_gap_of_terminal` reads the versions out of `db.drift_message`'s fixed opening clause.
SCHEMA_DRIFT_TERMINAL = "sisteer.registry.db.SchemaDrift:"
#: The registry's FOLD-ORDER refusal (`db.FoldOrderDrift`, a `SchemaDrift` by type: the database records a same-stamp fold order that is not the
#: code's -- `reducer.FOLD_ORDER_VERSION`), raised by the same claim path. Python's terminal line names the subclass, so the prefix is its own;
#: the kind is `schema_drift` all the same: held transient, never quarantined, never counted.
FOLD_ORDER_DRIFT_TERMINAL = "sisteer.registry.db.FoldOrderDrift:"
_SCHEMA_DRIFT_VERSIONS_RE = re.compile(r"is at schema version (?P<have>\d+)(?:, code expects (?P<want>\d+)| like the code)")
DISPATCH_RETRIES = "dispatch_retries.json"
#: The terminal line of a grader whose claim `pending.claim` refused as NOT FOLDED -- the run's or the judge's row has not folded yet (the
#: exception's qualified name under Python's traceback contract; both graders leave it uncaught on purpose: catching it would change the prefix
#: this daemon keys on). The `not_folded` kind: held transient like `schema_drift`, by PREFIX of the terminal line, never a substring mid-tail.
NOT_FOLDED_TERMINAL = "sisteer.registry.pending.ClaimNotFolded:"
#: pending.claim will also refuse with `ClaimPendingCorrection` -- a SUBCLASS of ClaimNotFolded -- while an identity correction of the run is
#: pending outside its own record files (a claim could otherwise bind a stale identity when the correction arrives as an adapter revision or a
#: run_patch); refused by name before anything is written, retry after the next harvester cycle. It is the `not_folded` kind exactly like its
#: parent: held transient, never a run failure, no quarantine, no abandonment, no family hold, no retryable count. The RULE is by TYPE, never
#: by the message and never by a name's text alone: a terminal line `sisteer.registry.pending.<Name>:` whose <Name> resolves by import (getattr
#: on the pending module) to ClaimNotFolded or a subclass is retry-later -- so every future subclass routes here by construction -- plus the one
#: class named in RETRY_LATER_CLAIM_REFUSAL (a single token to edit if upstream renames it), accepted by its prefix on a tree that does not
#: carry the class yet. The CLI (`registry pending --claim`) emits no traceback for either class: `claim refused: <Name>: <message>` on stderr,
#: exit 1 (cli.py names the class, so a held claim is told from a precheck refusal; no daemon path drives the CLI today). An earlier CLI branch
#: returned CLAIM_REFUSED_CLI_CLASS for EVERY line under the prefix before any type test (the red at
#: tests/test_grade_daemon_identity_changed.py:357), so `claim refused: ClaimPrecheckRefused: ...` (a ClaimHeld subclass) read retry-later /
#: not_folded against this function's own docstring and the pin that the CLI's own line is not a grader's terminal (`failed`). Now the CLI
#: branch reads the wrapped `<Name>:` and applies the SAME resolution as the module-prefixed branch (`_retry_later_class_name`): a retry-later
#: class -> CLAIM_REFUSED_CLI_CLASS (the tick's `claim_retry_later` key is unchanged); ClaimPrecheckRefused / ClaimHeld / an unknown name ->
#: None, and `dispatch_failure_kind` falls through to its existing rules. A CLI line carrying NO class name (not the CLI's shape; the
#: message-only pins) keeps the earlier reading: the same kind by the prefix.
#: The tick counts the runs so held per class (`claim_retry_later`).
RETRY_LATER_CLAIM_REFUSAL = "ClaimPendingCorrection"
NOT_FOLDED_TERMINAL_MODULE_PREFIX = "sisteer.registry.pending."
CLAIM_REFUSED_CLI_PREFIX = "claim refused: "
CLAIM_REFUSED_CLI_CLASS = "cli_claim_refused"
#: main's f0b2ed0c8 widened every multi-run grader's claim handler to `except (RW.P.ClaimHeld, RW.P.ClaimNotFolded) as e:` and each re-raises
#: its OWN wrapper of `--worklist registry: refusing — {e}` -- petri_run an uncaught DriverError (`DriverError: …` as a script,
#: `petri_run.DriverError: …` imported; rc 1), misalign_judge / served_capability_judge / grokmt_grade / grade_bon_live a bare SystemExit
#: (rc 1), grade_full22_native `[<target>] …` (SystemExit, rc 1; the vendored GRADER_DIFF.patch the same), grade_static_sampled /
#: grade_padded_static `REFUSING: [<run>] …` (Refuse, rc 2) -- so a ClaimNotFolded-class refusal no longer reaches the daemon as the
#: module-prefixed traceback the type rule classifies, and read `failed` (the quarantine path). The wrapper drops the class name, so the match
#: is on the MESSAGE texts (src/sisteer/registry/pending.py on main, cited by line), never on the class name or the wrapper alone, each with its
#: own retry class for the tick's `claim_retry_later`. ClaimHeld's wrapped message (`run … is claimed by session …`, pending.py:51) is not
#: retry-later-by-fold and keeps `failed`; an identity-change message under the same wrapper stays `identity_changed`.
WRAPPED_NOT_FOLDED_PHRASE = "is not folded yet; retry after the harvester folds it"                        # pending.py:62  ClaimNotFolded.__init__
WRAPPED_PENDING_CORRECTION_PHRASE = "an identity correction of this run is pending outside its own record files"   # pending.py:78  ClaimPendingCorrection.__init__
WRAPPED_NOT_PENDING_RE = re.compile(r"not pending for judge \S+ any more -- ")                           # pending.py:97  ClaimNotPending.__init__ (main; not on this tree yet)
WRAPPED_RETRY_LATER_CLASSES: Tuple[Tuple[str, object], ...] = (("wrapped_not_folded", WRAPPED_NOT_FOLDED_PHRASE),
                                                              ("wrapped_pending_correction", WRAPPED_PENDING_CORRECTION_PHRASE),
                                                              ("wrapped_not_pending", WRAPPED_NOT_PENDING_RE))
#: the graders' wrapper forms (f0b2ed0c8): an optional prefix -- `DriverError: ` (optionally module-qualified), `REFUSING: [<tag>] ` or `[<tag>] ` --
#: then the worklist's `--worklist registry: refusing — ` (em dash, or hyphens); the message follows
WRAPPED_REFUSAL_RE = re.compile(r"^(?:(?:[A-Za-z_][\w.]*\.)?DriverError: |REFUSING: \[(?P<rtag>[^\]]*)\] |\[(?P<btag>[^\]]*)\] )?--worklist registry: refusing (?:\u2014|-+) ")


def _retry_later_class_name(name: str) -> Optional[str]:
    """RETRYCLASSFIX: the retry-later resolution shared by both shapes of a claim refusal's terminal line -- ``name`` when it is ClaimNotFolded,
    the class the owner mailed (RETRY_LATER_CLAIM_REFUSAL), or resolves on the imported pending module to a ClaimNotFolded subclass (the type
    decides, never the name's text); None for a name that resolves to anything else (ClaimHeld, ClaimPrecheckRefused), to nothing, or is not an
    identifier."""
    if not name or not name.isidentifier():
        return None
    if name in ("ClaimNotFolded", RETRY_LATER_CLAIM_REFUSAL):
        return name
    base = getattr(P, "ClaimNotFolded", None)
    cls = getattr(P, name, None)
    if isinstance(base, type) and isinstance(cls, type) and issubclass(cls, base):
        return name
    return None


def wrapped_retry_later_class(terminal_line: str, group_tags: Optional[frozenset] = None) -> Optional[str]:
    """The retry class of a grader's WRAPPED claim refusal on its TERMINAL line -- `wrapped_not_folded` /
    `wrapped_pending_correction` / `wrapped_not_pending` when the line is one of the graders' `--worklist registry: refusing — <message>` wrappers
    (WRAPPED_REFUSAL_RE: the uncaught DriverError, the bare SystemExit, the `[<target>]`-tagged SystemExit, the `REFUSING: [<run>] ` Refuse) and
    <message> carries one of the three ClaimNotFolded-class texts pending.py emits (WRAPPED_RETRY_LATER_CLASSES); None for any other
    line: ClaimHeld's wrapped message, an identity-change refusal under the same wrapper (`identity_changed_terminal`'s business), a plain
    `REFUSING: [run] …`, the phrase outside a wrapper. A tagged wrapper whose tag is not one of ``group_tags`` (when given: the dispatch group's
    private run and run ids) is not this dispatch's refusal: None."""
    line = terminal_line or ""
    m = WRAPPED_REFUSAL_RE.match(line)
    if not m:
        return None
    tag = m.group("rtag") if m.group("rtag") is not None else m.group("btag")
    if tag is not None and group_tags is not None and tag not in group_tags:
        return None
    msg = line[m.end():]
    for cls, probe in WRAPPED_RETRY_LATER_CLASSES:
        if (probe.search(msg) is not None) if hasattr(probe, "search") else (str(probe) in msg):
            return cls
    return None


def retry_later_refusal_class(terminal_line: str, group_tags: Optional[frozenset] = None) -> Optional[str]:
    """The class of a retry-later claim refusal on the grader's TERMINAL line, else None: `sisteer.registry.pending.<Name>: ...` where <Name> is
    ClaimNotFolded, the class named in RETRY_LATER_CLAIM_REFUSAL, or any name that resolves on the imported pending module to a
    ClaimNotFolded subclass (the type decides, never the name's text; a name that resolves to anything else -- ClaimHeld, ClaimPrecheckRefused --
    or to nothing is None); the CLI's `claim refused: <Name>: ...` line is CLAIM_REFUSED_CLI_CLASS when its <Name> resolves the same way
    (a CLI-wrapped ClaimPrecheckRefused / ClaimHeld / unknown name is None, so `dispatch_failure_kind` reads the line by its other
    rules -- `failed` for the CLI's own line, as pinned), and a CLI line that carries no class name keeps the earlier reading
    (CLAIM_REFUSED_CLI_CLASS by the prefix). A grader's own `--worklist registry: refusing — <message>` wrapper (f0b2ed0c8) is
    classed by the MESSAGE (`wrapped_retry_later_class`: wrapped_not_folded / wrapped_pending_correction / wrapped_not_pending), never by
    the class name it dropped; ``group_tags`` applies the tag rule to the tagged wrappers."""
    line = terminal_line or ""
    if line.startswith(CLAIM_REFUSED_CLI_PREFIX):
        head, sep, _rest = line[len(CLAIM_REFUSED_CLI_PREFIX):].partition(":")
        name = head.strip()
        if sep and name.isidentifier():                   # the CLI's shape (cli.py): `claim refused: <Name>: <message>` -- the type decides
            return CLAIM_REFUSED_CLI_CLASS if _retry_later_class_name(name) else None
        return CLAIM_REFUSED_CLI_CLASS                    # no class name on the line (the message-only shape): the same kind by the prefix
    if not line.startswith(NOT_FOLDED_TERMINAL_MODULE_PREFIX):
        return wrapped_retry_later_class(line, group_tags)         # RETRYCLASSFIX2: the graders' own wrappers, by the owner's message
    return _retry_later_class_name(line[len(NOT_FOLDED_TERMINAL_MODULE_PREFIX):].split(":", 1)[0].strip())
#: consecutive ticks a run may stand refused as not folded before it is a `problem` (then every tick, uncapped: visible, never quarantined)
NOT_FOLDED_PROBLEM_TICKS = 2
#: The Petri grader's IDENTITY-CHANGE refusals -- petri_run's borrow /
#: filing rechecks raising `DriverError` when the registry moved under the grader between the binding and the filing ("anchor identity changed
#: between the binding and ...", "in-view run identity changed between the binding and the filing ...", "selected run dropped from the
#: judged view between the worklist read and the borrow ...") -- are a CORRECT, TRANSIENT answer, not a defect: the daemon re-selects the group
#: next tick with its copy kept (the `identity_changed` kind: pending, copy kept, no quarantine, no retryable count), one
#: `dispatch_identity_changed` per group naming the run ids and the refusal text, a `problem` when the same group refuses on consecutive ticks
#: (uncapped, like `not_folded`). Read from the TERMINAL line only: it must be the grader's DriverError line (bare under __main__, as the
#: dispatch template runs it; module-qualified from another entry point; optionally wrapping the refusal as `--worklist registry: refusing — ...`)
#: AND carry one of the refusals' stable phrases -- the same phrase buried mid-tail under another terminal line stays `failed`, and a
#: DriverError with another message is a real failure. The same class covers the source-cell refusals: "source changed during
#: binding/snapshot" (the bound `none` source re-scored between the binding and the copy) and "changed while it was being snapshotted".
#: The TERMINAL line may be the grader's DriverError line (petri_run wraps every
#: claim refusal as `DriverError: --worklist registry: refusing — ...`), the bare `--worklist registry: refusing — ...` message (the misalign
#: judge's SystemExit), the two dispatched STATIC graders' `Refuse` print -- `REFUSING: [<run>] --worklist registry: refusing — ...` (exit 2;
#: grade_static_sampled.py / grade_padded_static.py), admitted NARROWLY: the registry wrapper must follow the run tag -- or the uncaught
#: exception line of the precheck refusal classes (worklist.StaleRequiredJudge, pending.ClaimPrecheckRefused), which name themselves. Each must
#: carry one of the phrases (the exception line of those two classes is its own phrase); a line with none of these prefixes never matches, and
#: a `REFUSING:` line of any other kind is a real failure.
#: The second branch also admits grade_full22_native's `[<target>] ` tag before the worklist wrapper (its
#: `raise SystemExit(f"[{args.target}] --worklist registry: refusing — {e}")`), under the same group_tags rule as `REFUSING: [<run>] `
#: (`_REFUSING_TAG_RE` reads the tag of either tagged form); every other admitted form is unchanged
IDENTITY_CHANGED_TERMINAL_RE = re.compile(r"^(?:(?:[A-Za-z_][\w.]*\.)?(?:DriverError|StaleRequiredJudge|ClaimPrecheckRefused): (?:--worklist registry: refusing (?:\u2014|-+) )?"
                                          r"|(?:REFUSING: \[[^\]]*\] |\[[^\]]*\] )?--worklist registry: refusing (?:\u2014|-+) )")
IDENTITY_CHANGED_EXCEPTION_RE = re.compile(r"(?:StaleRequiredJudge|ClaimPrecheckRefused): ")
#: The `[<run>]` tag of a `REFUSING:` line is the grader's private run (grade_static_sampled:
#: its config's `private_run`; grade_padded_static: the padded run the generator named -- both the dispatch group's `private_run`, which is how
#: the daemon chose that config). A tag naming a run outside the group being classified is NOT this dispatch's refusal (a stale or foreign log
#: line, a grader reading another root): `failed`, never transient. The group's tags are threaded from the dispatch outcome (its private run
#: and its run ids); a caller without them (the pure classifier) accepts any tag.
_REFUSING_TAG_RE = re.compile(r"^(?:REFUSING: )?\[([^\]]*)\] ")           # The `[<tag>]` of either tagged wrapper form
IDENTITY_CHANGED_PHRASES = ("identity changed between the binding and",
                            "dropped from the judged view between the worklist read and the borrow",
                            # the source-cell refusals of the same class (the bound source re-scored or rewritten under the borrow)
                            "source changed during binding/snapshot",
                            "changed while it was being snapshotted",
                            # worklist.StaleRequiredJudge -- the run's designated judge moved between the pending read and the
                            # claim ("claim refused: run <id> now designates judge <current> (instrument <id>), the worklist row carried <old>");
                            # a quarantine would release on nothing a new designation changes
                            "now designates judge",
                            # the registry owner's distinct precheck refusal (pending.ClaimPrecheckRefused) -- its message's stable
                            # opening, and the class names as the CLI / a wrapper spell them: re-select, never a blind retry
                            "claim refused (precheck)", "ClaimPrecheckRefused", "StaleRequiredJudge")
#: consecutive ticks a group may stand refused on identity before it is a `problem` (then every tick, uncapped: visible, never quarantined)
IDENTITY_CHANGED_PROBLEM_TICKS = 2
#: An identity refusal that PERSISTS while the run stays pending re-dispatched every
#: tick -- mirror + grader start, no cap, no hold -- and dispatch is serial, so one stuck group could starve every other group indefinitely.
#: After IDENTITY_HOLD_AFTER refusals with an UNCHANGED fingerprint -- the registered identity of the group's runs AND of every run the
#: refusal names (IDENTITY_FINGERPRINT_FIELDS of the run row plus v_run_current's is_current / why), hashed at the first refusal and re-read
#: at every later one -- the GROUP is HELD, never the family: skipped before its mirror and claim, the copy kept, the `problem` still raised
#: every tick -- until the fingerprint changes (the registry moved: released, re-selected and dispatched) or the operator's release file
#: names the run ids (`<state-dir>/dispatch_identity_hold.release`: one run id per line or `*`, consumed like the quarantine's). The ledger
#: persists in `<state-dir>/dispatch_identity_hold.json`, so a re-exec grants no fresh dispatches.
IDENTITY_HOLD_AFTER = 3
DISPATCH_IDENTITY_HOLD_RELEASE = "dispatch_identity_hold.release"
DISPATCH_IDENTITY_LEDGER = "dispatch_identity_hold.json"
#: Every compared routing / identity field of the run row -- petri_run.BORROW_IN_VIEW_IDENTITY_FIELDS
#: (run_id, eval_id, producer_object_key_legacy, object_id, campaign, store_uri, substrate, n_items_generated, required_judge_id, instrument_id,
#: instrument_judge_id, model_id, model_revision, store_completion_artifact_uri, store_completion_sha256: the run id keys the fact and the two
#: judge fields are read from the instrument) plus the daemon's own; pinned by tests/test_grade_daemon_identity_changed.py against the
#: module's constant (importing petri_run here is not cheap). The daemon's code identity rides beside them (identity_fingerprint_facts).
IDENTITY_FINGERPRINT_FIELDS = ("lifecycle_state", "store_completion_sha256", "store_completion_artifact_uri", "store_uri", "n_items_generated",
                               "instrument_id", "object_id", "producer_object_key_legacy", "eval_id", "campaign", "model_id", "model_revision",
                               "substrate", "view_legacy", "superseded_by_run_id", "judge_id", "n_items_judged")
_RUN_ID_RE = re.compile(r"\b[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+:[0-9a-f]{12}:[0-9a-f]{16}\b")
#: THE DERIVED-ROWS LEDGER. The misalign grader files its COMPLETE judgments and then derives the
#: capeval rows (its rows step); a failed rows step exits 1 -- but the daemon then only QUARANTINED the dispatch, and the
#: run, covered by a complete judgment, had left v_pending_judgment: the next tick did not select it, quarantine_filter dropped the entry
#: as "settled by another route", and the derived rows stayed missing for ever. So a dispatch of a DERIVED_ROWS_FAMILIES group writes one
#: DURABLE work item per run (`dispatch_rows_pending.json`: run id, the judgment ids it bound, the dispatch's retained served-identity
#: snapshot, the grader command, the mirrored store directories, an attempt count) BEFORE its grader starts, and clears it only when the
#: rows step SUCCEEDED (the grader's own clean exit, or a later retry's). Every tick's `rows_retry_pass` drives the open items from the
#: RETAINED snapshot -- the grader re-run `--rows-only` over the same root and snapshot, under the group's store locks -- until success,
#: bounded at ROWS_ATTEMPT_CAP attempts (the grader's own counts as the first), then the item enters the named terminal state
#: `rows_exhausted`: kept in the ledger, counted in every tick summary with a `problem`, never dropped silently; the operator re-arms it
#: through `dispatch_rows_pending.release` (run ids or `*`). While an item is open the stores it reads are KEPT: the settlement treats the
#: dispatch as copy-keeping and the prune pass defers the group. A dispatch counts as `dispatched` (complete) only when its items are
#: cleared; a grader that filed and then failed is `dispatch_rows_failed`, not a quarantine (the quarantine is for a run still pending).
#: A run of such a dispatch with NO complete judgment of its own in the spool drops its item (nothing to derive rows from: the ordinary
#: failure path re-judges it).
#: The ledger is GLOBAL to the state dir while the per-store locks cover store directories only, so
#: every change goes through rows_update -- a cross-process flock on `<ledger>.lock` held across one read -> modify -> atomic replace
#: (writer-unique temp names) -- and a settlement after a long subprocess re-reads the live ledger and touches only its own items (an
#: instance that wrote back its pre-subprocess snapshot deleted another instance's freshly opened item). And a copy an OPEN item references
#: (its run id, or a recorded mirrored store directory; any state, awaiting_grader included) is never retired by mirror_settle -- the
#: claim reconciliation of a daemon that died after its grader filed but before the fold used to retire exactly the judged.jsonl the
#: adopted retry needed; the tick runs rows_retry_pass ahead of that reconciliation.
#: A dead daemon's `awaiting_grader` item is settled only once its grader's COMPLETION is
#: established -- by locks, as the claim reconciliation does: the per-dispatch lock the daemon takes before the grader and the grader
#: inherits (`<dispatch log base>.<pid>.lock`, dispatch_lock_for), plus the item's recorded store locks; while either is held the grader
#: still runs and the item is kept (`rows_awaiting_grader`), never dropped under a grader that is about to file.
DISPATCH_ROWS = "dispatch_rows_pending.json"
DISPATCH_ROWS_RELEASE = "dispatch_rows_pending.release"
ROWS_ATTEMPT_CAP = 3
ROWS_STATE_AWAITING = "awaiting_grader"
ROWS_STATE_PENDING = "pending"
ROWS_STATE_EXHAUSTED = "rows_exhausted"
#: the families whose grader derives rows AFTER filing its judgments (the misalign judge's rows step over its results root)
DERIVED_ROWS_FAMILIES = ("agentic",)
GRADER_LOCK_TIMEOUT_ENV = "SISTEER_REGISTRY_LOCK_TIMEOUT_S"
GRADER_LOCK_TIMEOUT_S_DEFAULT = 300.0


def terminal_exception_line(tail: str) -> str:
    """The last non-blank line of the grader's output — the terminal exception under Python's traceback contract."""
    for ln in reversed((tail or "").splitlines()):
        if ln.strip():
            return ln.strip()
    return ""


def credential_rejected(tail: str) -> bool:
    """Whether the grader's output carries the API's credential rejection (`CREDENTIAL_TAIL_RE`: 401 authentication_error / invalid
    x-api-key / AuthenticationError, 403 PermissionDeniedError) -- anywhere in the tail, by design (the constant's note)."""
    return bool(CREDENTIAL_TAIL_RE.search(tail or ""))


def credential_org_of_tail(tail: str) -> Tuple[str, bool]:
    """`(org, named)`: the org the grader's tail names last (`_CREDENTIAL_ORG_RE`), else the vault's default org with `named=False`."""
    found = _CREDENTIAL_ORG_RE.findall(tail or "")
    return (found[-1], True) if found else (CREDENTIAL_DEFAULT_ORG, False)


def unscored_line_is_terminal(last: str) -> bool:
    """Whether a JB1 UNSCORED terminal line carries evidence its failures were NOT shedding -- embedded judge-client counts with
    terminal > rate_limited, or a failure-types histogram naming a 4xx other than 429."""
    t = _JUDGE_COUNT_RE["terminal"].search(last or ""); r = _JUDGE_COUNT_RE["rate_limited"].search(last or "")
    if t and r and int(t.group(1)) > int(r.group(1)):
        return True
    return bool(_FAILURE_TYPES_4XX_RE.search(last or ""))


def dispatch_failure_kind(rc, tail: str, group_tags: Optional[frozenset] = None) -> str:
    """'ok' on rc 0; 'credential' when the TAIL carries the API's credential rejection (`credential_rejected`: 401 / 403 -- the family is held,
    one problem, nothing quarantined, nothing counted); 'not_folded' when the TERMINAL line is the registry's claim refusal for an unfolded
    run or judge row (`NOT_FOLDED_TERMINAL`, by prefix: held transient, never quarantined, never counted); 'schema_drift' when it is
    the registry's schema refusal (`SCHEMA_DRIFT_TERMINAL`, by prefix: the same); 'retryable' when it is the registry lock
    contention (`RETRYABLE_TERMINAL`, by prefix) or the live judge's bounded-retry shortfall (`RETRYABLE_TERMINAL_SUBSTRINGS`, on that last line
    only -- and only when that line's own counts do not say the failures were terminal, `unscored_line_is_terminal`, else 'failed');
    'identity_changed' when it is the Petri grader's identity-change refusal (`identity_changed_terminal`: held transient like not_folded);
    else 'failed'. ``group_tags``: the dispatch group's private run and run ids -- and the target its
    grader was dispatched with (`target_for`), the tag grade_full22_native wraps its refusals in -- a `REFUSING: [<run>]` / `[<target>]` line
    whose tag is not one of them is `failed`."""
    if rc == 0:
        return "ok"
    if credential_rejected(tail):
        return "credential"
    last = terminal_exception_line(tail)
    if retry_later_refusal_class(last, group_tags) is not None:   # by type, the CLI's refusal, or a grader's wrapper by the owner's message
        return "not_folded"
    if last.startswith(SCHEMA_DRIFT_TERMINAL) or last.startswith(FOLD_ORDER_DRIFT_TERMINAL):
        return "schema_drift"
    if identity_changed_terminal(last, group_tags):
        return "identity_changed"
    if any(last.startswith(m) for m in RETRYABLE_TERMINAL):
        return "retryable"
    if any(m in last for m in RETRYABLE_TERMINAL_SUBSTRINGS):
        return "failed" if unscored_line_is_terminal(last) else "retryable"
    return "failed"


def identity_changed_terminal(last: str, group_tags: Optional[frozenset] = None) -> bool:
    """Whether the grader's TERMINAL line is an identity-change refusal -- one of the prefixes
    `IDENTITY_CHANGED_TERMINAL_RE` admits (the DriverError line, the bare, `REFUSING: [<run>] `-tagged or `[<target>] `-tagged
    `--worklist registry: refusing — ` wrapper, the precheck refusal classes' exception line) whose message carries one of
    `IDENTITY_CHANGED_PHRASES`; the exception line of
    worklist.StaleRequiredJudge / pending.ClaimPrecheckRefused is its own phrase. A `REFUSING:` line's `[<run>]` tag must be one of
    ``group_tags`` when they are given (the dispatch group's private run and run ids): a foreign tag is not this dispatch's refusal."""
    m = IDENTITY_CHANGED_TERMINAL_RE.match(last or "")
    if not m:
        return False
    tag = _REFUSING_TAG_RE.match(m.group(0))
    if tag is not None and group_tags is not None and tag.group(1) not in group_tags:
        return False
    if IDENTITY_CHANGED_EXCEPTION_RE.search(m.group(0)):
        return True
    return any(p in last[m.end():] for p in IDENTITY_CHANGED_PHRASES)


def schema_gap_of_terminal(last: str) -> Dict[str, object]:
    """The schema versions a grader's `SchemaDrift` terminal line names -- `{have, want, shape_only}` from `db.drift_message`'s fixed opening
    clause (`is at schema version H, code expects W` / `is at schema version H like the code, but its shape differs`); None values when the
    line does not carry them (the event still names the refusal by its terminal line)."""
    m = _SCHEMA_DRIFT_VERSIONS_RE.search(last or "")
    if not m:
        return {"have": None, "want": None, "shape_only": None}
    have = int(m.group("have"))
    want = int(m.group("want")) if m.group("want") is not None else have
    return {"have": have, "want": want, "shape_only": m.group("want") is None}


#: `pending.claim` / `pending.release` fold only their own record and REFUSE by name --
#: `pending.ClaimNotFolded` -- a claim whose run or judge row has not folded yet (nothing written; retry after the harvester folds it). The
#: runs so refused, run id -> {since_utc, ticks, family, model_key, message, last_utc, dispatch_log}, per process: the `not_folded` dispatch
#: kind is held transient like `schema_drift` (pending, copy kept, no quarantine, no retryable count); a run at NOT_FOLDED_PROBLEM_TICKS
#: consecutive refusals is a `problem` every tick it stands (uncapped: a run whose row never folds must become visible, never quarantined);
#: cleared (`dispatch_not_folded_cleared`) when the run dispatches with another outcome or leaves the pending set.
_NOT_FOLDED: Dict[str, Dict[str, object]] = {}


def not_folded_clear(state_dir: Optional[str], run_ids: Iterable[str], why: str, family: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for rid in run_ids:
        was = _NOT_FOLDED.pop(str(rid), None)
        if was is None:
            continue
        out.append(str(rid))
        log_event(state_dir, {"event": "dispatch_not_folded_cleared", "run_id": str(rid), "family": family or was.get("family"), "why": why, "ticks": was.get("ticks"),
                              "since_utc": was.get("since_utc"), "was_message": was.get("message"),
                              "note": "the run's claim had been refused as not folded (dispatch_not_folded) and now " + why})
    return out


#: The runs whose grader refused on an identity change, run id -> {since_utc, ticks, family, model_key, message, last_utc, dispatch_log},
#: per process -- the `identity_changed` kind's standing ledger, the twin of `_NOT_FOLDED`: a group at IDENTITY_CHANGED_PROBLEM_TICKS consecutive
#: refusals is a `problem` every tick it stands (uncapped, never quarantined); cleared (`dispatch_identity_changed_cleared`) when the run dispatches
#: with another outcome or leaves the pending set
_IDENTITY_CHANGED: Dict[str, Dict[str, object]] = {}


def identity_changed_clear(state_dir: Optional[str], run_ids: Iterable[str], why: str, family: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for rid in run_ids:
        was = _IDENTITY_CHANGED.pop(str(rid), None)
        if was is None:
            continue
        out.append(str(rid))
        log_event(state_dir, {"event": "dispatch_identity_changed_cleared", "run_id": str(rid), "family": family or was.get("family"), "why": why,
                              "ticks": was.get("ticks"), "since_utc": was.get("since_utc"), "was_message": was.get("message"),
                              "note": "the run's grader had refused on an identity change (dispatch_identity_changed) and now " + why})
    if out:
        identity_ledger_save(state_dir)
    return out


def _identity_ledger_path(state_dir: str) -> str:
    return os.path.join(state_dir, DISPATCH_IDENTITY_LEDGER)


_IDENTITY_LEDGER_LOADED: set = set()


def identity_ledger_load(state_dir: Optional[str]) -> None:
    """The identity ledger persists across re-execs -- read ONCE per process per state dir into `_IDENTITY_CHANGED` (an entry the
    process already holds wins)."""
    if not state_dir or state_dir in _IDENTITY_LEDGER_LOADED:
        return
    _IDENTITY_LEDGER_LOADED.add(state_dir)
    try:
        with open(_identity_ledger_path(state_dir), encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return
    for rid, ent in (doc.get("runs") or {}).items() if isinstance(doc, dict) else []:
        if isinstance(ent, dict):
            _IDENTITY_CHANGED.setdefault(str(rid), dict(ent))


def identity_ledger_save(state_dir: Optional[str]) -> None:
    if not state_dir:
        return
    os.makedirs(state_dir, exist_ok=True)
    tmp = _identity_ledger_path(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"runs": _IDENTITY_CHANGED, "utc": utc_now(), "hold_after": IDENTITY_HOLD_AFTER}, fh, indent=2, sort_keys=True, default=str); fh.write("\n")
    os.replace(tmp, _identity_ledger_path(state_dir))


def identity_fingerprint_facts(registry_root: str, run_ids: Sequence[str], message: Optional[str]) -> Dict[str, object]:
    """The registered identity the refusal depends on -- for the group's runs AND every run id the refusal text
    names (the anchor / in-view run whose identity, currency or completion moved): IDENTITY_FINGERPRINT_FIELDS of the run row, the run's
    instrument's designated judge (`instrument_judge_id`), v_run_current's is_current / why / selection_state -- AND the daemon's own code
    identity (`daemon_code_sha`: a fixed grader or daemon lands as a new sha, so a group held on a bug the fix repairs is released by the
    re-exec, once, instead of waiting for a human). Read now; hashed by :func:`identity_fingerprint`."""
    ids = sorted(set(str(r) for r in run_ids) | set(_RUN_ID_RE.findall(message or "")))
    facts: Dict[str, object] = {"daemon_code_sha": IMPORTED_CODE_SHA, "fields": list(IDENTITY_FINGERPRINT_FIELDS), "runs": []}
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            for rid in ids:
                row = conn.execute("SELECT * FROM run WHERE run_id = ?", (rid,)).fetchone()
                run = {k: row[k] for k in row.keys() if k in IDENTITY_FINGERPRINT_FIELDS} if row is not None else None
                ij = conn.execute("SELECT i.judge_id FROM run r LEFT JOIN instrument i ON i.instrument_id = r.instrument_id WHERE r.run_id = ?", (rid,)).fetchone()
                try:
                    cur = conn.execute("SELECT * FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()
                    current = ({k: cur[k] for k in cur.keys() if k in ("is_current", "why", "selection_state")} if cur is not None else None)
                except sqlite3.Error:
                    current = "unreadable"
                facts["runs"].append({"run_id": rid, "run": run, "instrument_judge_id": (ij[0] if ij is not None else None), "current": current})
    except (sqlite3.Error, OSError) as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def identity_fingerprint(registry_root: str, run_ids: Sequence[str], message: Optional[str]) -> str:
    """The 16-hex digest of :func:`identity_fingerprint_facts`; a change in any fact -- the registry, or this daemon's code -- having moved
    releases a held group."""
    return hashlib.sha256(json.dumps(identity_fingerprint_facts(registry_root, run_ids, message), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _identity_release_requests(state_dir: Optional[str], consume: bool) -> Optional[set]:
    """The identity hold's release file (`DISPATCH_IDENTITY_HOLD_RELEASE`): run ids, or {'*'}; consumed when read; None without a file."""
    if not state_dir:
        return None
    path = os.path.join(state_dir, DISPATCH_IDENTITY_HOLD_RELEASE)
    try:
        with open(path, encoding="utf-8") as fh:
            ids = {ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")}
    except OSError:
        return None
    if consume:
        try:
            os.unlink(path)
        except OSError:
            pass
    return ids


def identity_hold_release(state_dir: Optional[str], registry_root: str, run_ids: Sequence[str], held_ids: Sequence[str],
                          asked: Optional[set]) -> Optional[str]:
    """Whether a HELD group may dispatch again: ``'operator'`` when the release file names one of its runs (or `*`) -- the standing count
    starts over; ``'fingerprint changed'`` when the registered identity moved since the hold -- the count starts over at the new
    fingerprint; None while it stays held. Releasing clears the hold flags and saves the ledger."""
    why = None
    if asked and (("*" in asked) or any(rid in asked for rid in held_ids)):
        why = "operator"
    else:
        ent0 = _IDENTITY_CHANGED.get(str(held_ids[0]), {})
        fp_now = identity_fingerprint(registry_root, run_ids, str(ent0.get("message") or ""))
        if fp_now != ent0.get("fingerprint"):
            why = "fingerprint changed"
    if why is None:
        return None
    for rid in held_ids:
        ent = _IDENTITY_CHANGED.get(str(rid))
        if ent is not None:
            ent.update({"held": False, "released_utc": utc_now(), "released_why": why, "n_unchanged": 0})
            if why == "fingerprint changed":
                ent.pop("fingerprint", None)
    identity_ledger_save(state_dir)
    return why


def _retries_path(state_dir: str) -> str:
    return os.path.join(state_dir, DISPATCH_RETRIES)


def _spool_judgment_records_since(registry_root: str, campaign: Optional[str], run_id: str, dispatch_utc: Optional[str]) -> List[Dict[str, object]]:
    """Every `judgment` record of the run on the registry spool written at or after `dispatch_utc`, WHATEVER its status --
    `[{status, judgment_id, written_utc}]` in file order: the same walk and the same durable boundary as the shared reader
    `pending.spool_complete_judgments` (which keeps the complete ones alone): a record not yet folded still counts as written; an unreadable
    file is skipped, never fatal. `campaign` None reads the run id's campaign prefix."""
    out: List[Dict[str, object]] = []
    d = os.path.join(RH.records_dir(registry_root), str(campaign or run_id.split(":", 1)[0]), RI.run_dir_name(run_id))
    if not os.path.isdir(d):
        return out
    import gzip
    for f in sorted(os.listdir(d)):
        if not (f.endswith(".jsonl") or f.endswith(".jsonl.gz")):
            continue
        opener = gzip.open if f.endswith(".gz") else open
        try:
            with opener(os.path.join(d, f), "rt", encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        rec = json.loads(ln)
                    except ValueError:
                        continue
                    pay = rec.get("payload") if isinstance(rec, dict) else None
                    if not (isinstance(pay, dict) and rec.get("record_type") == "judgment"):
                        continue
                    written = str(rec.get("written_utc") or "")
                    if dispatch_utc and written < dispatch_utc:
                        continue
                    out.append({"status": str(pay.get("status") or ""), "judgment_id": str(pay.get("judgment_id") or ""), "written_utc": written})
        except (OSError, EOFError):
            continue
    return out


def dispatch_progress(registry_root: Optional[str], campaign: Optional[str], group: Sequence[Mapping[str, object]],
                      dispatch_utc: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """What THIS dispatch did for each run of its group, read from the registry spool (the registry owner's ruling of 2026-10-13 23:29 on
    the retryable cap): `filed` -- the runs it filed a COMPLETE judgment for (the spooled hold's own criterion: a complete `judgment` record
    written at or after `dispatch_utc`); `attempted` -- the runs whose attempt it began and did not complete: a `partial` judgment record
    written at or after `dispatch_utc`, the scope's close when a grader's attempt ends short (the BoN leg's UNSCORED SystemExit, any
    exception inside the scope); `unreached` -- the rest of the group, for which no attempt began (a bare claim is the worklist's
    bookkeeping before the first API call, not an attempt, and a same-session re-claim writes no record at all). None when there is no
    registry root to read, in which case `retryable_count` charges every run as before the ruling."""
    if not registry_root:
        return None
    out: Dict[str, List[str]] = {"filed": [], "attempted": [], "unreached": []}
    for r in group:
        rid = str(r["run_id"])
        statuses = {x["status"] for x in _spool_judgment_records_since(registry_root, campaign or r.get("campaign"), rid, dispatch_utc)}  # type: ignore[index]
        if "complete" in statuses:
            out["filed"].append(rid)
        elif "partial" in statuses:
            out["attempted"].append(rid)
        else:
            out["unreached"].append(rid)
    return out


def retryable_count(state_dir: Optional[str], run_ids: Sequence[str], *, failed: bool, progress: Optional[Mapping[str, Sequence[str]]] = None) -> int:
    """Consecutive retryable failures per run; returns the group's max. Any other outcome (`failed` False) clears every run of the group.
    A retryable failure is charged by what the dispatch DID (the registry owner's ruling of 2026-10-13 23:29; `progress` from
    `dispatch_progress`): a run it filed a complete judgment for RESETS to 0 -- progress, not a consecutive failure; a run it attempted and
    left incomplete increments; a run it never reached is not charged (its counter stands as it was). A dispatch that left NO judgment
    record for any run of the group -- the registry lock refused the first claim, a grader dead before any scope closed -- is a consecutive
    failure of the WHOLE group and every run is charged: the LockHeld class the cap was written for stays bounded. With no `progress` (no
    registry root to read) every run is charged, as before the ruling."""
    if not state_dir:
        return 1 if failed else 0
    try:
        with open(_retries_path(state_dir), encoding="utf-8") as fh:
            doc = json.load(fh); counts = dict(doc.get("runs") or {}) if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        counts = {}
    if failed:
        filed = {str(x) for x in (progress or {}).get("filed", ())}
        attempted = {str(x) for x in (progress or {}).get("attempted", ())}
        if progress is None or not (filed or attempted):
            attempted = {str(rid) for rid in run_ids}          # no per-run evidence: the dispatch as a whole failed, every run charged
        for rid in run_ids:
            k = str(rid)
            if k in filed:
                counts[k] = 0
            elif k in attempted:
                counts[k] = counts.get(k, 0) + 1
    else:
        for rid in run_ids:
            counts[str(rid)] = 0
    counts = {k: v for k, v in counts.items() if v}
    os.makedirs(state_dir, exist_ok=True)
    tmp = _retries_path(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"runs": counts, "utc": utc_now()}, fh, indent=2, sort_keys=True); fh.write("\n")
    os.replace(tmp, _retries_path(state_dir))
    return max((counts.get(str(rid), 0) for rid in run_ids), default=0)

def daemon_code_sha() -> str:
    """This daemon's code identity, read FRESH on every call (a cached value never sees a re-pointed checkout, so the
    quarantine's "new daemon sha" release never fires): the git HEAD of the checkout this file runs from, suffixed `+<12 hex>` of the file's
    digest when the working file differs from that commit; the digest alone (`file:<sha256>`) for a copy outside git."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(__file__, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    sha = ""
    try:
        r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            sha = r.stdout.strip()
            d = subprocess.run(["git", "-C", here, "diff", "--quiet", "HEAD", "--", os.path.abspath(__file__)], capture_output=True, text=True, timeout=30)
            if d.returncode != 0:
                sha += "+" + digest[:12]
    except (OSError, subprocess.SubprocessError):
        sha = ""
    return sha or f"file:{digest}"


#: The code this PROCESS imported. `check_code_drift` compares it with the checkout's HEAD before every tick: the unit runs from a detached
#: checkout that is re-pointed at landings, so a difference means the daemon's own code is stale — it says so (`code_drift`) and re-executes
#: itself on the new code (os.execv: same pid, same unit, same argv; the graders and the Petri drive it launched are unaffected — they are
#: children of the pid, their pid markers and lock descriptions live on). A `--once` run only reports the drift.
IMPORTED_CODE_SHA = daemon_code_sha()


def check_code_drift(args) -> Optional[str]:
    """None when the checkout's HEAD is the imported code; else the current sha, after logging `code_drift`."""
    current = daemon_code_sha()
    if current == IMPORTED_CODE_SHA:
        return None
    log_event(getattr(args, "state_dir", None), {"event": "code_drift", "imported_sha": IMPORTED_CODE_SHA, "current_sha": current,
                                                   "action": "report only (--once)" if getattr(args, "once", False) else "re-exec on the new code",
                                                   "note": "the checkout this daemon runs from was re-pointed: quarantine releases and every later tick must run the new code"})
    return current


def reexec_self() -> None:
    """Replace this process with a fresh interpreter on the same argv (the new code)."""
    sys.stdout.flush(); sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _spooled_path(state_dir: str) -> str:
    return os.path.join(state_dir, DISPATCH_SPOOLED)


def spooled_load(state_dir: Optional[str]) -> Dict[str, dict]:
    if not state_dir:
        return {}
    try:
        with open(_spooled_path(state_dir), encoding="utf-8") as fh:
            doc = json.load(fh)
        runs = doc.get("runs") if isinstance(doc, dict) else None
        return {str(k): dict(v) for k, v in runs.items()} if isinstance(runs, dict) else {}
    except (OSError, ValueError):
        return {}


def spooled_save(state_dir: str, runs: Mapping[str, Mapping[str, object]]) -> None:
    os.makedirs(state_dir, exist_ok=True)
    tmp = _spooled_path(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"runs": {k: dict(v) for k, v in sorted(runs.items())}, "utc": utc_now()}, fh, indent=2, sort_keys=True, default=str); fh.write("\n")
    os.replace(tmp, _spooled_path(state_dir))


def spooled_hold_add(state_dir: Optional[str], *, group: Sequence[Mapping[str, object]], family: str, model_key: str, campaign: Optional[str], log_path: str,
                     registry_root: Optional[str] = None, dispatch_utc: Optional[str] = None) -> None:
    """After a CLEAN dispatch: every run of the group whose grader FILED a complete judgment in the spool is held until its fold is observed —
    the hold bound to those judgment ids (records written at or after `dispatch_utc`) and to the store generation the grader read; a run with
    no such record gets no hold (the grader exited clean without filing it: the dead-grader reclaim path stands) and is named."""
    if not state_dir:
        return
    runs = spooled_load(state_dir); now = utc_now(); held: List[str] = []; unfiled: List[str] = []
    gens: Dict[str, str] = {}
    if registry_root:
        try:
            with DB.reading(DB.db_path(registry_root)) as conn:
                for r in group:
                    gens[str(r["run_id"])] = store_generation(conn, str(r["run_id"]), r)
        except Exception as exc:  # noqa: BLE001 — no registry answer: the row's own fields stand in (as the quarantine does)
            log_event(state_dir, {"event": "spooled_generation_deferred", "error": f"{type(exc).__name__}: {exc}"})
    for r in group:
        rid = str(r["run_id"]); camp = campaign or r.get("campaign")
        jids = [j["judgment_id"] for j in _spool_complete_judgments(registry_root, camp, rid) if not dispatch_utc or str(j.get("written_utc") or "") >= dispatch_utc] if registry_root else []
        if not jids:
            unfiled.append(rid); runs.pop(rid, None); continue
        runs[rid] = {"run_id": rid, "family": family, "model_key": model_key, "campaign": camp, "spooled_utc": now, "dispatch_utc": dispatch_utc or now,
                     "dispatch_log": log_path, "daemon_sha": IMPORTED_CODE_SHA, "judgment_ids": sorted(set(jids)),
                     "store_generation": gens.get(rid, store_generation(None, rid, r)),
                     "identity_binding": identity_binding_of(r)}        # the identity the grader was dispatched against
        held.append(rid)
    spooled_save(state_dir, runs)
    if held:
        log_event(state_dir, {"event": "dispatch_spooled", "family": family, "model_key": model_key, "n": len(held), "run_ids": held,
                              "judgment_ids": {rid: runs[rid]["judgment_ids"] for rid in held},
                              "note": "the grader exited clean: these judgments are in the spool; the runs are held from re-dispatch until the harvester's fold shows "
                                      "in the registry (the hold is bound to these judgment ids and the store generation they graded)"})
    if unfiled:
        log_event(state_dir, {"event": "dispatch_spooled_unfiled", "family": family, "model_key": model_key, "n": len(unfiled), "run_ids": unfiled,
                              "note": "the grader exited clean but filed no complete judgment for these runs in the spool: no hold (the own-session reclaim rule decides)"})


def _spool_complete_judgments(registry_root: Optional[str], campaign: Optional[str], run_id: str) -> List[Dict[str, object]]:
    """Every complete `judgment` record for the run in the registry spool (the grader filed before exiting): [{judgment_id, judge_id,
    n_items_judged, claimed_by_session, written_utc}], in file order -- the shared reader (`pending.spool_complete_judgments`), so the
    daemon and the drivers' registry worklist read one definition of "filed"."""
    return P.spool_complete_judgments(registry_root, campaign, run_id) if registry_root else []


def _spool_has_complete_judgment(registry_root: str, campaign: Optional[str], run_id: str) -> bool:
    """A complete `judgment` record for the run sits in the registry spool (the grader filed before exiting) — the LEGACY proof, read only
    for ledger entries written before the hold was bound to its dispatch's judgment ids."""
    return bool(_spool_complete_judgments(registry_root, campaign, run_id))


def _spooled_release_requests(state_dir: str, consume: bool) -> set:
    """`dispatch_spooled.release`: run ids (or `*`) the operator releases from the spooled hold (an unresolved spool refusal), consumed when read."""
    path = os.path.join(state_dir, DISPATCH_SPOOLED_RELEASE)
    try:
        with open(path, encoding="utf-8") as fh:
            ids = {ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")}
    except OSError:
        return set()
    if consume:
        try:
            os.unlink(path)
        except OSError:
            pass
    return ids


def _quarantine_folded_short(state_dir: str, rid: str, ent: Mapping[str, object], *, folded: Sequence[Mapping[str, object]], pend: Mapping[str, object],
                             verdict: Optional[str] = None, verdict_label: str = "identity_stale") -> str:
    """A bound judgment FOLDED and the run still pends: a NAMED coverage defect, quarantined (the quarantine's release paths are the
    controlled retry); returns the defect line. ``verdict``: why an identity_stale re-pend -- or, under ``verdict_label`` "binding change",
    a changed binding -- was NOT accepted as the explanation (a wrong-judge completion, missing or unchanged binding evidence),
    named in the defect. The record carries the run's identity binding
    at filing time and each folded judgment's judge and binding, so the quarantine can release on a verified change of that binding."""
    jl = "; ".join(f"judgment {j['judgment_id']} (run {j.get('run_id')}, judge {j.get('judge_id')}) status={j.get('status')} n_items_judged={j.get('n_items_judged')}"
                   for j in folded)
    why = (f"coverage_short: {jl}; the run still pends for its instrument judge after the fold (pending_reason={pend.get('pending_reason')}, "
           f"required_judge_id={pend.get('required_judge_id')}, required_coverage={pend.get('required_coverage')}, items_missing={pend.get('items_missing')}) — "
           "the grader's judgment does not cover the store; a re-dispatch would re-judge the same store to the same shortfall"
           + (f"; {verdict_label} not accepted: {verdict}" if verdict else ""))
    runs = quarantine_load(state_dir)
    runs[rid] = {"run_id": rid, "family": ent.get("family"), "model_key": ent.get("model_key"), "campaign": ent.get("campaign"), "eval_id": pend.get("eval_id"),
                 "rc": "coverage_short", "stderr_tail": why, "daemon_sha": IMPORTED_CODE_SHA, "store_generation": str(ent.get("store_generation") or ""),
                 "cmd": "", "dispatch_log": ent.get("dispatch_log"), "utc": utc_now(), "judgment_ids": list(ent.get("judgment_ids") or []),
                 "identity_binding": identity_binding_of(pend),
                 "judgment_bindings": [{"judgment_id": str(j.get("judgment_id")), "run_id": j.get("run_id"), "judge_id": j.get("judge_id"),
                                        "binding": j.get("stated_binding") or ent.get("identity_binding")} for j in folded]}
    quarantine_save(state_dir, runs)
    return why


def spooled_rebuild(state_dir: Optional[str], registry_root: str, rows: Sequence[Dict[str, object]], *, dry_run: bool = False) -> List[str]:
    """A `reclaim` row -- claimed by this daemon's graders' session with no grader running -- whose claim's COMPLETE
    judgment is already on the spool was FILED by a grader that exited before the daemon bound its hold (the daemon died between the grader's
    exit and `spooled_hold_add`; `mirror_reconcile_claims` settles the mirror only). It is not a dead grader's run: re-dispatching it would
    judge it again under the same id. The hold is re-established here, bound to that judgment id and the store generation, so `spooled_filter`
    holds the run until the fold shows (or releases it into the quarantine when the fold leaves it pending). A reclaim row with no such record
    stays reclaimable. Returns the rebuilt run ids; on a dry run nothing is written (the caller holds them from its report by hand)."""
    if not state_dir:
        return []
    runs = spooled_load(state_dir); now = utc_now(); rebuilt: List[str] = []
    cands = [r for r in rows if r.get("reclaim") and r.get("claim_judgment_id") and str(r["run_id"]) not in runs]
    if not cands:
        return []
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            for r in cands:
                rid = str(r["run_id"]); camp = r.get("campaign"); jid = str(r["claim_judgment_id"])
                if not any(j["judgment_id"] == jid for j in _spool_complete_judgments(registry_root, camp, rid)):
                    continue
                runs[rid] = {"run_id": rid, "family": str(r.get("family") or ""), "model_key": str(r.get("view_legacy") or r.get("model_id") or ""),
                             "campaign": camp, "spooled_utc": now, "dispatch_utc": str(r.get("claim_utc") or now), "dispatch_log": "", "daemon_sha": IMPORTED_CODE_SHA,
                             "judgment_ids": [jid], "store_generation": store_generation(conn, rid, r), "rebuilt_utc": now}
                rebuilt.append(rid)
    except Exception as exc:  # noqa: BLE001 -- no registry answer: nothing rebuilt this tick (the reclaim rows are reported, not dispatched blindly: spooled_filter runs next)
        log_event(state_dir, {"event": "spooled_rebuild_deferred", "error": f"{type(exc).__name__}: {exc}"})
        return []
    if rebuilt:
        if not dry_run:
            spooled_save(state_dir, runs)
        log_event(state_dir, {"event": "spooled_hold_rebuilt", "n": len(rebuilt), "run_ids": sorted(rebuilt), "judgment_ids": {rid: runs[rid]["judgment_ids"] for rid in rebuilt},
                              "dry_run": dry_run,
                              "note": "claimed by this daemon's graders' session and the claim's complete judgment is already on the spool: the grader filed before the "
                                      "hold was bound; held from re-dispatch until the fold shows in the registry (never a reclaim)"})
    return rebuilt


def spooled_filter(state_dir: Optional[str], registry_root: str, rows: Sequence[Dict[str, object]], *, dry_run: bool = False
                   ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Clear every hold whose run has left its instrument judge's pending view (folded), drop every hold whose spool carries no complete
    judgment (the grader died before filing — reclaimable), raise a `problem` for holds older than SPOOLED_HOLD_WARN_S, then split `rows`
    into (eligible, held). A bound judgment that FOLDED while the run still pends is a named coverage defect (quarantined) -- unless the run
    pends as IDENTITY_STALE_REASON (its identity was corrected under the judgment, which now counts for nothing), or its identity
    BINDING changed since the dispatch whatever the reason says (a required-judge move lists the run as awaiting_second_judge,
    never identity_stale) -- each VERIFIED against the folded judgments (this run's, the judge required at dispatch, a differing binding): then
    the hold is cleared and the run re-dispatched under the current identity."""
    runs = spooled_load(state_dir)
    if not runs or not state_dir:
        return list(rows), []
    cleared: List[Tuple[str, str]] = []; dropped: List[Tuple[str, str]] = []; released: List[Tuple[str, str]] = []
    asked = _spooled_release_requests(state_dir, consume=not dry_run)
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            for rid, ent in list(runs.items()):
                camp = ent.get("campaign")
                if "*" in asked or rid in asked:
                    dropped.append((rid, "release_file")); continue
                if not run_pending(conn, rid, camp):
                    cleared.append((rid, "folded")); continue
                bound = [str(j) for j in (ent.get("judgment_ids") or [])]
                if not bound:                                            # a ledger entry from before the binding: the legacy proof, logged
                    if not _spool_has_complete_judgment(registry_root, camp, rid):
                        dropped.append((rid, "legacy_no_complete_judgment"))
                    else:
                        log_event(state_dir, {"event": "spooled_hold_legacy", "run_id": rid, "note": "hold written before the judgment binding: held on the directory's complete judgment until it clears"})
                    continue
                if str(ent.get("store_generation") or "") and store_generation(conn, rid, None) != str(ent.get("store_generation") or ""):
                    cleared.append((rid, "store_regenerated")); continue
                qmarks = ",".join("?" for _ in bound)
                # the grader's CLAIM folds at once under the same judgment id (status in_progress); the fold that matters is its COMPLETION
                # record reaching the registry (complete / partial — or a later state the harvester moved it to)
                folded = judgment_rows_with_binding(conn, bound)      # judge, status, counts and the judgment's stated binding when the row has one
                if folded:                                               # this dispatch's judgment reached the registry, yet the run still pends
                    pend = (P.pending(conn, None, camp, run_id=rid) or [{}])[0]
                    verdict: Optional[str] = None; verdict_label = "identity_stale"
                    reason = str(pend.get("pending_reason") or "")
                    current_b = identity_binding_of(pend); bound_b = ent.get("identity_binding")
                    # DAEMONQUARFIX (the range audit on 6e486c9e4): the judge the folded judgments must be by is the one REQUIRED AT DISPATCH -- the
                    # hold's bound binding names it; a hold with no binding (legacy, or rebuilt from the spool) falls back to the current required judge
                    dispatch_judge = str((bound_b or {}).get("required_judge_id") or pend.get("required_judge_id") or "")
                    folded_ev = [{"judgment_id": str(j["judgment_id"]), "run_id": j.get("run_id"), "judge_id": j.get("judge_id"), "status": j.get("status"),
                                  "n_items_judged": j.get("n_items_judged"), "stated_binding": j.get("stated_binding")} for j in folded]
                    foreign_ids = sorted({str(j.get("run_id")) for j in folded if str(j.get("run_id") or "") != rid})
                    if reason == IDENTITY_STALE_REASON:
                        # The bound judgment folded and the run still pends because the run identity it was
                        # bound to was CORRECTED -- the registry's window re-pends the same row as `identity_stale` and the judgment counts
                        # for nothing (its row stands). Not a coverage defect of the grader's: the hold is cleared and the run re-dispatched; the
                        # fresh claim (pending.claim mints a current-bound one) judges it against the corrected identity. Never the quarantine.
                        # The reason is existential over ANY old judgment of the required judge and the hold binds EVERY complete
                        # judgment the grader spooled, so the clear is VERIFIED first: every folded judgment must be the required judge's and made
                        # under a binding that differs from the run's current identity; a wrong-judge completion or missing / unchanged binding
                        # evidence falls back to the coverage_short path with the verdict named.
                        # The verdict also requires every folded judgment to be THIS run's (a foreign-run completion
                        # written through this run's directory and bound by the hold is never this run's fold); the event names each folded judgment's run
                        ok, verdict = identity_stale_clear_verdict(folded, run_id=rid, required_judge=dispatch_judge, current=current_b, bound=bound_b)
                        log_event(state_dir, {"event": "spooled_hold_identity_stale", "run_id": rid, "judgment_ids": [str(j["judgment_id"]) for j in folded],
                                              "folded": folded_ev, "foreign_run_ids": foreign_ids,
                                              "required_judge_id": pend.get("required_judge_id"), "bound_required_judge_id": (bound_b or {}).get("required_judge_id"),
                                              "current_binding": current_b, "bound_binding": bound_b, "verified": ok, "verdict": verdict,
                                              "pending_reason": pend.get("pending_reason"), "family": ent.get("family"), "store_generation": ent.get("store_generation"),
                                              "note": ("this dispatch's judgment folded while the run still pends because the run identity it was bound to was "
                                                       "corrected: the judgment counts for nothing; cleared for re-dispatch under the current identity (never a "
                                                       "coverage defect, never the quarantine)" if ok else
                                                       "the run pends as identity_stale but the folded judgment is not verified as the stale-bound one for the "
                                                       "required judge: the coverage_short path decides (quarantine), the verdict names why")})
                        if ok:
                            cleared.append((rid, "identity_stale_rebind")); continue
                    elif identity_binding_differs(bound_b, current_b):
                        # DAEMONQUARFIX: the run's identity binding CHANGED since the dispatch -- the required judge moved (J1 -> J2: the view lists the
                        # run for J2 as `awaiting_second_judge`, J1's completion being another judge's there, never `identity_stale`) or the instrument
                        # did under a reason the identity_stale branch does not read. The fold of a dispatch made under the OLD binding counts for
                        # nothing: cleared for re-dispatch under the current identity once VERIFIED (this run's judgments, by the judge required at
                        # dispatch, under a binding that differs from the current one); refused, the coverage_short path decides with the verdict
                        # named. Before this the fold was quarantined coverage_short with the CORRECTED binding recorded, so the quarantine's
                        # identity_rebound release could never fire and the new judge was never dispatched without an operator release.
                        verdict_label = "binding change"
                        ok, verdict = binding_changed_clear_verdict(folded, run_id=rid, current=current_b, bound=bound_b)
                        log_event(state_dir, {"event": "spooled_hold_binding_changed", "run_id": rid, "judgment_ids": [str(j["judgment_id"]) for j in folded],
                                              "folded": folded_ev, "foreign_run_ids": foreign_ids,
                                              "bound_required_judge_id": (bound_b or {}).get("required_judge_id"), "required_judge_id": pend.get("required_judge_id"),
                                              "bound_binding": bound_b, "current_binding": current_b, "verified": ok, "verdict": verdict,
                                              "pending_reason": pend.get("pending_reason"), "family": ent.get("family"), "store_generation": ent.get("store_generation"),
                                              "note": ("the run's identity binding (instrument / required judge) changed since this dispatch and its judgment folded "
                                                       "under the old one: it counts for nothing under the current identity; cleared for re-dispatch against it "
                                                       "(never a coverage defect, never the quarantine)" if ok else
                                                       "the run's identity binding changed since this dispatch but the folded judgment is not verified as the "
                                                       "dispatch's own (this run's, the judge required at dispatch, the old binding): the coverage_short path "
                                                       "decides (quarantine), the verdict names why")})
                        if ok:
                            cleared.append((rid, "binding_changed_rebind")); continue
                    why = (_quarantine_folded_short(state_dir, rid, ent, folded=folded, pend=pend, verdict=verdict, verdict_label=verdict_label) if not dry_run
                           else "coverage_short (dry run: not quarantined)" + (f"; {verdict_label} not accepted: {verdict}" if verdict else ""))
                    released.append((rid, why)); continue
                in_spool = {j["judgment_id"] for j in _spool_complete_judgments(registry_root, camp, rid)}
                if not set(bound) <= in_spool:
                    dropped.append((rid, "bound_judgment_not_in_spool"))
    except Exception as exc:  # noqa: BLE001 — no registry answer: every hold stands this tick
        log_event(state_dir, {"event": "spooled_check_deferred", "error": f"{type(exc).__name__}: {exc}"})
    for rid, why in cleared:
        ent = runs.pop(rid, {})
        log_event(state_dir, {"event": "spooled_hold_cleared", "run_id": rid, "reason": why, "family": ent.get("family"), "spooled_utc": ent.get("spooled_utc"),
                              "judgment_ids": ent.get("judgment_ids"),
                              "note": {"folded": "the fold shows in the registry: the run left its judge's pending view",
                                       "store_regenerated": "the store's generation changed since the dispatch: the graded store is gone; the run is dispatchable again",
                                       "identity_stale_rebind": "this dispatch's judgment folded under a since-corrected run identity and counts for nothing (g41-fix): "
                                                                f"the run pends as {IDENTITY_STALE_REASON} and is dispatchable again under the current identity",
                                       "binding_changed_rebind": "this dispatch's judgment folded under a run identity (instrument / required judge) since corrected and "
                                                                 "counts for nothing (DAEMONQUARFIX): the run pends for its current required judge and is dispatchable "
                                                                 "again under the current identity"}.get(why, why)})
    for rid, why in dropped:
        ent = runs.pop(rid, {})
        log_event(state_dir, {"event": "spooled_hold_dropped", "run_id": rid, "reason": why, "family": ent.get("family"), "spooled_utc": ent.get("spooled_utc"),
                              "judgment_ids": ent.get("judgment_ids"),
                              "note": {"release_file": f"released by the operator through {DISPATCH_SPOOLED_RELEASE}",
                                       "bound_judgment_not_in_spool": "this dispatch's complete judgment record is no longer in the run's spool directory and never folded: nothing to wait for — reclaimable",
                                       "legacy_no_complete_judgment": "no complete judgment record in the run's spool directory: the grader did not file before exiting — reclaimable"}[why]})
    for rid, why in released:
        ent = runs.pop(rid, {})
        log_event(state_dir, {"event": "spooled_hold_released", "run_id": rid, "reason": "folded_still_pending", "family": ent.get("family"), "judgment_ids": ent.get("judgment_ids"),
                              "quarantined": not dry_run, "defect": why,
                              "note": "this dispatch's judgment folded and the run still pends: quarantined under the named coverage defect (never re-dispatched blindly; "
                                      f"release with a new daemon sha, a re-filed store, or {DISPATCH_QUARANTINE_RELEASE})"})
    if (cleared or dropped or released) and not dry_run:
        spooled_save(state_dir, runs)
    held = [r for r in rows if str(r["run_id"]) in runs]
    eligible = [r for r in rows if str(r["run_id"]) not in runs]
    if held:
        log_event(state_dir, {"event": "spooled_hold", "n": len(held), "run_ids": sorted(str(r["run_id"]) for r in held),
                              "note": "judgments spooled by a clean grader, not yet folded: held from re-dispatch (fold_mode: harvester)"})
        stale = [rid for rid in (str(r["run_id"]) for r in held) if _age_s(runs.get(rid, {}).get("spooled_utc")) > SPOOLED_HOLD_WARN_S]
        if stale:
            log_event(state_dir, {"event": "problem", "kind": "spooled_not_folded", "n": len(stale), "run_ids": sorted(stale), "ledger": _spooled_path(state_dir),
                                  "note": f"judgments spooled more than {int(SPOOLED_HOLD_WARN_S)} s ago and still not folded: check the harvester's cycles and its refusals (the hold stands)"})
    return eligible, held


def _age_s(utc: Optional[str]) -> float:
    if not utc:
        return 0.0
    try:
        from datetime import datetime, timezone
        t = datetime.strptime(str(utc)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds())
    except ValueError:
        return 0.0


def _quarantine_path(state_dir: str) -> str:
    return os.path.join(state_dir, DISPATCH_QUARANTINE)


def quarantine_load(state_dir: Optional[str]) -> Dict[str, dict]:
    if not state_dir:
        return {}
    try:
        with open(_quarantine_path(state_dir), encoding="utf-8") as fh:
            doc = json.load(fh)
        runs = doc.get("runs") if isinstance(doc, dict) else None
        return {str(k): dict(v) for k, v in runs.items()} if isinstance(runs, dict) else {}
    except (OSError, ValueError):
        return {}


def quarantine_save(state_dir: str, runs: Mapping[str, Mapping[str, object]]) -> None:
    os.makedirs(state_dir, exist_ok=True)
    tmp = _quarantine_path(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"runs": {k: dict(v) for k, v in sorted(runs.items())}, "utc": utc_now()}, fh, indent=2, sort_keys=True, default=str); fh.write("\n")
    os.replace(tmp, _quarantine_path(state_dir))


def store_generation(conn, run_id: str, row: Optional[Mapping[str, object]] = None) -> str:
    """The store's identity as the registry records it: `store_uri | n_items_generated | store_completion_sha256 | generated_utc` from the
    run table (the worklist row's `store_uri` / `n_items_generated` when the table has no row). A closer that re-files the store, or a
    regenerated store, changes it; a grader failure does not."""
    hit = conn.execute("SELECT store_uri, n_items_generated, store_completion_sha256, generated_utc FROM run WHERE run_id = ?", (run_id,)).fetchone() if conn is not None else None
    if hit is None:
        return f"{(row or {}).get('store_uri') or ''}|{(row or {}).get('n_items_generated')}||"
    return "|".join("" if v is None else str(v) for v in hit)


def _dispatch_log_path(state_dir: Optional[str], family: str, model_key: str) -> str:
    log_dir = os.path.join(state_dir, DISPATCH_LOG_DIR) if state_dir else tempfile.mkdtemp(prefix="grade-daemon-dispatch-")
    os.makedirs(log_dir, exist_ok=True)
    stamp = re.sub(r"[^0-9A-Za-z]", "", utc_now())
    return os.path.join(log_dir, f"{stamp}_{re.sub(r'[^0-9A-Za-z_.-]', '_', family)}_{re.sub(r'[^0-9A-Za-z_.-]', '_', model_key)}.log")


def _tail(path: str, n: int = DISPATCH_TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END); size = fh.tell(); fh.seek(max(0, size - n * 2))
            return fh.read().decode("utf-8", errors="replace")[-n:]
    except OSError:
        return ""


def quarantine_add(state_dir: Optional[str], registry_root: str, *, group: Sequence[Mapping[str, object]], family: str, model_key: str,
                   campaign: Optional[str], rc, cmd: Sequence[str], log_path: str) -> None:
    """After a group's dispatch returned non-zero: record every run of the group (the same grader served them all) with the failure's
    evidence and the cause markers the release reads, and say so once (`dispatch_failed`)."""
    tail = _tail(log_path)
    sha = IMPORTED_CODE_SHA          # the code that dispatched and classified this failure (never the checkout's HEAD,
    run_ids = [str(r["run_id"]) for r in group]   # which may already be re-pointed while a long grader runs; the re-exec on the new code releases it)
    log_event(state_dir, {"event": "dispatch_failed", "family": family, "model_key": model_key, "campaign": campaign, "n": len(group), "run_ids": run_ids,
                          "rc": rc, "stderr_tail": tail, "terminal_exception": terminal_exception_line(tail), "daemon_sha": sha, "dispatch_log": log_path,
                          "note": "quarantined: not mirrored or dispatched again until the daemon's code sha or the store's generation changes, "
                                  f"or {DISPATCH_QUARANTINE_RELEASE} names the run (or `*`)"})
    if not state_dir:
        return
    gens: Dict[str, str] = {}
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            for r in group:
                gens[str(r["run_id"])] = store_generation(conn, str(r["run_id"]), r)
    except Exception as exc:  # noqa: BLE001 — no registry answer: the row's own fields stand in
        log_event(state_dir, {"event": "quarantine_generation_deferred", "error": f"{type(exc).__name__}: {exc}", "run_ids": run_ids})
        gens = {str(r["run_id"]): store_generation(None, str(r["run_id"]), r) for r in group}
    runs = quarantine_load(state_dir)
    now = utc_now()
    for r in group:
        rid = str(r["run_id"])
        runs[rid] = {"run_id": rid, "family": family, "model_key": model_key, "campaign": campaign, "eval_id": r.get("eval_id"), "rc": rc, "stderr_tail": tail,
                     "daemon_sha": sha, "store_generation": gens.get(rid, ""), "cmd": " ".join(shlex.quote(c) for c in cmd), "dispatch_log": log_path, "utc": now,
                     "identity_binding": identity_binding_of(r)}        # the identity the failed dispatch was made against
    quarantine_save(state_dir, runs)


def _rows_path(state_dir: str) -> str:
    return os.path.join(state_dir, DISPATCH_ROWS)


def rows_load(state_dir: Optional[str]) -> Dict[str, dict]:
    if not state_dir:
        return {}
    try:
        with open(_rows_path(state_dir), encoding="utf-8") as fh:
            doc = json.load(fh)
        runs = doc.get("runs") if isinstance(doc, dict) else None
        return {str(k): dict(v) for k, v in runs.items()} if isinstance(runs, dict) else {}
    except (OSError, ValueError):
        return {}


def rows_save(state_dir: str, runs: Mapping[str, Mapping[str, object]]) -> None:
    """The ledger written atomically through a WRITER-UNIQUE temp name (pid + a monotonic nonce: two instances once shared one
    `.tmp`). Changes go through rows_update, which holds the ledger lock around the read that fed `runs`."""
    os.makedirs(state_dir, exist_ok=True)
    tmp = f"{_rows_path(state_dir)}.{os.getpid()}.{time.monotonic_ns():x}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"runs": {k: dict(v) for k, v in sorted(runs.items())}, "utc": utc_now()}, fh, indent=2, sort_keys=True, default=str); fh.write("\n")
    os.replace(tmp, _rows_path(state_dir))


@contextlib.contextmanager
def rows_ledger_locked(state_dir: str):
    """An exclusive, BLOCKING flock on `<ledger>.lock` beside the ledger, held across exactly ONE read ->
    modify -> replace (rows_update) -- never across a grader or a rows-only subprocess, so the wait is a file write long. The per-store
    locks exclude instances per store directory; the ledger is one file for every dispatch of the state dir, and two instances with
    non-overlapping groups write it concurrently."""
    os.makedirs(state_dir, exist_ok=True)
    fd = os.open(_rows_path(state_dir) + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def rows_update(state_dir: Optional[str], mutate):
    """The ONE way the ledger changes: under the ledger lock, load the CURRENT file, apply ``mutate(runs)`` (edits the dict in
    place; whatever it returns is returned), save. A writer therefore never writes back a snapshot it loaded before a long step: a
    settlement after an hours-long grader or rows-only run re-reads the ledger another instance may have changed meanwhile and touches
    only its own items. Without a state dir the mutation runs over an empty dict and nothing is written."""
    if not state_dir:
        return mutate({})
    with rows_ledger_locked(state_dir):
        runs = rows_load(state_dir)
        res = mutate(runs)
        rows_save(state_dir, runs)
    return res


def _rows_release_requests(state_dir: str, consume: bool) -> set:
    """`dispatch_rows_pending.release`: run ids (one per line) or `*`; consumed unless a dry run."""
    path = os.path.join(state_dir, DISPATCH_ROWS_RELEASE)
    try:
        with open(path, encoding="utf-8") as fh:
            asked = {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}
    except OSError:
        return set()
    if consume:
        try:
            os.remove(path)
        except OSError:
            pass
    return asked


def dispatch_lock_for(log_path: str) -> str:
    """The dispatch's own lock file, beside its log and keyed like its identity snapshot (`<log base>.<pid>.lock`)."""
    base = log_path[:-len(".log")] if log_path.endswith(".log") else log_path
    return f"{base}.{os.getpid()}.lock"


def dispatch_lock_take(path: str) -> Optional[int]:
    """An exclusive flock on the dispatch's lock file, taken by the daemon right before its grader starts and INHERITED by the grader
    (pass_fds) -- so the lock is free only once the daemon AND the grader are gone, whether or not the judged root was mirrored (an unmirrored
    root has no store locks). Released by closing the fd (never LOCK_UN, which would release the grader's inherited copy too). Returns the fd,
    or None when the lock could not be taken (a fresh per-dispatch path is never contended; the item then relies on the store locks)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd); return None
    return fd


def dispatch_in_flight(ent: Mapping[str, object]) -> List[str]:
    """The locks of the item's dispatch that are STILL HELD -- its dispatch lock (dispatch_lock_take: the daemon's, inherited by the
    grader) and the recorded mirrored store directories' locks (mirror_locks, as the claim reconciliation checks them). Non-empty means the
    grader (or its daemon) is still running: the item's outcome is not established and it must be kept, not dropped and not adopted."""
    busy: List[str] = []
    lp = str(ent.get("dispatch_lock") or "")
    if lp and os.path.exists(lp):
        try:
            fd = os.open(lp, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                busy.append(lp)
            finally:
                os.close(fd)
        except OSError:
            pass
    dirs = [str(d) for d in ent.get("dirs") or [] if d]
    if dirs:
        with mirror_locks(dirs) as lk:
            busy.extend(lk.busy)
    return busy


def rows_items_open(state_dir: Optional[str], *, group: Sequence[Mapping[str, object]], family: str, model_key: str, campaign: Optional[str],
                    store_root: Optional[str], snapshot_dir: Optional[str], cmd: Sequence[str], dirs: Sequence[str], dispatch_utc: str, log_path: str,
                    dispatch_lock: Optional[str] = None) -> List[str]:
    """BEFORE the grader (and so before its rows step), one DURABLE work item per run of a DERIVED_ROWS_FAMILIES group --
    `awaiting_grader`, attempts 0, naming this daemon (pid + start) so a later instance can tell a dead daemon's item from a live one's --
    cleared only by rows_items_settle (a clean grader) or rows_retry_pass (a clean retry). Written through rows_update (under
    the ledger lock, over the current file). Returns the run ids opened."""
    if not state_dir or family not in DERIVED_ROWS_FAMILIES:
        return []
    now = utc_now(); pid = os.getpid(); start = _proc_start(pid); opened: List[str] = []

    def mutate(runs: Dict[str, dict]) -> None:
        for r in group:
            rid = str(r["run_id"])
            runs[rid] = {"run_id": rid, "family": family, "model_key": model_key, "campaign": campaign or r.get("campaign"), "store_root": store_root,
                         "snapshot_dir": snapshot_dir, "cmd": [str(c) for c in cmd], "dirs": sorted({str(d) for d in dirs if d}), "dispatch_utc": dispatch_utc,
                         "dispatch_log": log_path, "dispatch_lock": dispatch_lock, "judgment_ids": [], "attempts": 0, "state": ROWS_STATE_AWAITING, "opened_utc": now, "last_utc": now,
                         "last_error": None, "last_log": None, "daemon_pid": pid, "daemon_start": start, "daemon_sha": IMPORTED_CODE_SHA}
            opened.append(rid)
    rows_update(state_dir, mutate)
    log_event(state_dir, {"event": "rows_opened", "family": family, "model_key": model_key, "run_ids": opened, "snapshot_dir": snapshot_dir, "ledger": _rows_path(state_dir),
                          "note": "the derived-rows work item of this dispatch, written before its grader: cleared only when the rows step succeeds"})
    return opened


def _rows_filed_judgments(registry_root: Optional[str], campaign: Optional[str], run_id: str, dispatch_utc: Optional[str]) -> List[str]:
    """The complete judgment ids THIS dispatch filed for the run (spool records written at or after `dispatch_utc`: the spooled hold's own
    criterion); empty when the grader filed nothing for it."""
    if not registry_root:
        return []
    return sorted({str(j["judgment_id"]) for j in _spool_complete_judgments(registry_root, campaign, run_id)
                   if not dispatch_utc or str(j.get("written_utc") or "") >= dispatch_utc})


def rows_items_settle(state_dir: Optional[str], registry_root: Optional[str], *, group: Sequence[Mapping[str, object]], family: str, campaign: Optional[str],
                      rc, tail: str, dispatch_utc: str, log_path: str) -> Dict[str, List[str]]:
    """AFTER the grader exited -- {cleared, pending, dropped}. rc 0: the rows step succeeded (a failed one exits 1),
    every item is cleared. Otherwise each run whose complete judgment THIS dispatch filed keeps its item -- `pending`, attempt 1 (the
    grader's own), the judgment ids bound, the terminal line as its error -- for rows_retry_pass; a run with no such judgment drops its item
    (the ordinary failure path -- quarantine or retry -- re-judges it). Through rows_update -- the ledger is re-read under its
    lock AFTER the (hours-long) grader and only this group's items are touched; an item another instance opened meanwhile is untouched."""
    out: Dict[str, List[str]] = {"cleared": [], "pending": [], "dropped": []}
    if not state_dir or family not in DERIVED_ROWS_FAMILIES:
        return out
    now = utc_now(); term = terminal_exception_line(tail)

    def mutate(runs: Dict[str, dict]) -> None:
        for r in group:
            rid = str(r["run_id"]); ent = runs.get(rid)
            if ent is None:
                continue
            if rc == 0:
                runs.pop(rid); out["cleared"].append(rid); continue
            jids = _rows_filed_judgments(registry_root, campaign or r.get("campaign"), rid, dispatch_utc)
            if not jids:
                runs.pop(rid); out["dropped"].append(rid); continue
            ent.update({"state": ROWS_STATE_PENDING, "attempts": 1, "judgment_ids": jids, "last_utc": now, "last_error": term, "last_log": log_path})
            out["pending"].append(rid)
    rows_update(state_dir, mutate)
    if out["cleared"]:
        log_event(state_dir, {"event": "rows_cleared", "family": family, "run_ids": out["cleared"], "by": "grader", "attempt": 1,
                              "note": "the grader's rows step succeeded: the dispatch is complete"})
    if out["dropped"]:
        log_event(state_dir, {"event": "rows_dropped", "family": family, "run_ids": out["dropped"],
                              "note": "the grader filed no complete judgment for these runs: no rows to derive; the failure path re-judges them"})
    return out


def rows_retry_pass(args, summary: Dict[str, object], *, held: bool, held_families: Optional[Mapping[str, str]] = None) -> None:
    """Each tick, FIRST (ahead of the claim reconciliation, whose settlement of a dead daemon's dispatch retires
    still-pending copies) -- the ledger's open items. Phase 1, one locked read-modify-write (rows_update): an operator release re-arms the
    named (or every) item; an `awaiting_grader` item of a DEAD daemon is settled from the spool (filed -> pending; not -> dropped) while a
    live instance's is left alone; the `pending` items under ROWS_ATTEMPT_CAP are the pass's work, grouped per dispatch (the items sharing
    a snapshot). Phase 2, per group and OUTSIDE the ledger lock: the recorded grader command re-run `--rows-only` over the same root and
    RETAINED snapshot, under the group's store locks (a busy lock defers the group), its output in a `<family>-rows` dispatch log. Phase 3,
    one locked read-modify-write per group: the ledger is RE-READ and only this group's items, as they stand now, are settled -- success
    clears them (`rows_cleared`); a failure counts an attempt and at the cap enters the named terminal state `rows_exhausted` (kept,
    counted, a `problem` at the tick's end; never dropped); an item another instance cleared or re-armed meanwhile is left as it stands
    (`rows_retry_stale`). Under the operator hold nothing is retried (a hold dispatches no grader); a dry run reports and retries nothing.
    The summary carries rows_retried / rows_cleared here and rows_pending / rows_exhausted from the tick's closing recount.
    An item of a family under `dispatch.hold.<family>` (`held_families`) is not retried either -- kept as it stands, named in a
    `rows_retry_held` event of kind `family`."""
    state_dir = args.state_dir
    if not state_dir or not rows_load(state_dir):
        return
    fam_held = dict(held_families or {})
    if args.dry_run:
        runs = rows_load(state_dir)
        pend = sorted(rid for rid, e in runs.items() if e.get("state") == ROWS_STATE_PENDING and int(e.get("attempts") or 0) < ROWS_ATTEMPT_CAP)
        fam_skip = sorted(rid for rid in pend if str(runs[rid].get("family")) in fam_held)
        if fam_skip:
            log_event(state_dir, {"event": "rows_retry_held", "hold": "family", "run_ids": fam_skip, "families": sorted({str(runs[r].get("family")) for r in fam_skip}),
                                  "dry_run": True, "note": "operator family hold: no rows retry for these items"})
            pend = [rid for rid in pend if rid not in set(fam_skip)]
        if pend:
            log_event(state_dir, {"event": "rows_retry", "run_ids": pend, "dry_run": True, "attempts": {rid: runs[rid].get("attempts") for rid in pend}})
        return
    asked = _rows_release_requests(state_dir, consume=True)
    # A dead daemon's `awaiting_grader` item whose grader STILL RUNS -- its dispatch lock or its store
    # locks held (dispatch_in_flight) -- has no outcome yet: kept as it is, neither dropped (it may file in a minute) nor adopted. Checked
    # outside the ledger lock (the lock probes are non-blocking; a grader finishing between the probe and the write keeps its item one tick).
    in_flight: Dict[str, List[str]] = {}
    for rid, ent in rows_load(state_dir).items():
        if ent.get("state") != ROWS_STATE_AWAITING:
            continue
        pid, start = ent.get("daemon_pid"), ent.get("daemon_start")
        if isinstance(pid, int) and _proc_alive(pid, start if isinstance(start, int) else None):
            continue
        busy = dispatch_in_flight(ent)
        if busy:
            in_flight[rid] = busy
    if in_flight:
        log_event(state_dir, {"event": "rows_awaiting_grader", "run_ids": sorted(in_flight), "locks": in_flight,
                              "note": "the dispatching daemon is gone but its grader still holds the dispatch's locks: the rows item is kept as it is until the grader exits "
                                      "(then adopted if it filed, dropped if it did not)"})

    def prepare(runs: Dict[str, dict]):
        notes: List[Tuple[str, str, object]] = []
        for rid, ent in list(runs.items()):
            if ("*" in asked or rid in asked) and ent.get("state") in (ROWS_STATE_PENDING, ROWS_STATE_EXHAUSTED):
                ent.update({"attempts": 0, "state": ROWS_STATE_PENDING, "released_utc": utc_now()}); notes.append(("rows_released", rid, None))
            if ent.get("state") == ROWS_STATE_AWAITING:
                pid, start = ent.get("daemon_pid"), ent.get("daemon_start")
                if isinstance(pid, int) and _proc_alive(pid, start if isinstance(start, int) else None):
                    continue                                # a live instance's dispatch: its business
                if rid in in_flight:
                    continue                                # its grader still runs -- no outcome yet
                jids = _rows_filed_judgments(args.registry_root, ent.get("campaign") or args.campaign, rid, ent.get("dispatch_utc"))
                if jids:
                    ent.update({"state": ROWS_STATE_PENDING, "attempts": max(1, int(ent.get("attempts") or 0)), "judgment_ids": jids, "last_utc": utc_now(),
                                "last_error": "the dispatching daemon died under its grader after the judgments were filed"})
                    notes.append(("rows_adopted", rid, jids))
                else:
                    runs.pop(rid); notes.append(("rows_dropped", rid, ent.get("family")))
        work = [dict(e) for e in runs.values() if e.get("state") == ROWS_STATE_PENDING and int(e.get("attempts") or 0) < ROWS_ATTEMPT_CAP]
        return notes, work
    notes, work = rows_update(state_dir, prepare)
    for kind, rid, extra in notes:
        if kind == "rows_released":
            log_event(state_dir, {"event": kind, "run_id": rid, "note": f"re-armed by {DISPATCH_ROWS_RELEASE}: attempts reset, retried from the retained snapshot"})
        elif kind == "rows_adopted":
            log_event(state_dir, {"event": kind, "run_id": rid, "judgment_ids": extra, "note": "a dead daemon's dispatch filed this run's judgment: its rows item is pending here"})
        else:
            log_event(state_dir, {"event": kind, "family": extra, "run_ids": [rid], "note": "a dead daemon's dispatch filed no complete judgment for this run: no rows to derive"})
    groups: Dict[str, List[dict]] = {}
    for e in work:
        groups.setdefault(str(e.get("snapshot_dir") or e.get("store_root") or e["run_id"]), []).append(e)
    retried = cleared = 0
    for key, items in sorted(groups.items()):
        rids = sorted(str(e["run_id"]) for e in items); first = items[0]
        if held:
            log_event(state_dir, {"event": "rows_retry_held", "run_ids": rids, "snapshot_dir": key, "note": "operator dispatch hold: no rows retry this tick"})
            continue
        fam_now = family_holds(state_dir)                               # the CURRENT holds before every spawn -- a hold written during an
        fam = str(first.get("family"))                                  # earlier retry of this pass stops the later ones; the start-of-tick snapshot cannot
        if fam in fam_now:                                              # the family's hold covers its rows retries too
            log_event(state_dir, {"event": "rows_retry_held", "hold": "family", "family": first.get("family"), "reason": fam_now[fam], "mid_tick": fam not in fam_held,
                                  "run_ids": rids, "snapshot_dir": key, "note": "operator family hold: no rows retry for this family this tick; the items stand"})
            continue
        cmd = [str(c) for c in first.get("cmd") or []] + ["--rows-only"]
        dirs = sorted({str(d) for e in items for d in e.get("dirs") or [] if d})
        with mirror_locks(dirs) as lk:
            if lk.busy:
                log_event(state_dir, {"event": "rows_retry_deferred", "run_ids": rids, "dirs": lk.busy, "note": "another instance holds these store directories: retried on a later tick"})
                continue
            log_path = _dispatch_log_path(state_dir, f"{first.get('family')}-rows", str(first.get("model_key") or ""))
            attempt_n = int(first.get("attempts") or 0) + 1                # this attempt's ordinal (the grader's own was 1)
            log_event(state_dir, {"event": "rows_retry", "run_ids": rids, "snapshot_dir": key, "attempt": attempt_n, "cap": ROWS_ATTEMPT_CAP,
                                  "cmd": " ".join(shlex.quote(c) for c in cmd), "log": log_path, "dry_run": False})
            grader_env = dict(os.environ)
            grader_env.setdefault(GRADER_LOCK_TIMEOUT_ENV, str(getattr(args, "grader_lock_timeout_s", GRADER_LOCK_TIMEOUT_S_DEFAULT)))
            try:
                with open(log_path, "ab") as log_fh:
                    rc = subprocess.run(cmd, cwd=args.repo, timeout=args.driver_timeout_s, pass_fds=tuple(lk.fds), stdout=log_fh, stderr=subprocess.STDOUT, env=grader_env).returncode
            except (OSError, subprocess.SubprocessError) as exc:
                rc = f"error: {exc}"
        now = utc_now(); tail = _tail(log_path) if rc != 0 else ""; term = terminal_exception_line(tail)

        def settle(runs: Dict[str, dict]) -> Dict[str, List[str]]:
            res: Dict[str, List[str]] = {"cleared": [], "failed": [], "exhausted": [], "stale": []}
            for rid in rids:
                ent = runs.get(rid)
                if ent is None or ent.get("state") != ROWS_STATE_PENDING:
                    res["stale"].append(rid); continue      # cleared or re-armed by another instance meanwhile: left as it stands
                if rc == 0:
                    runs.pop(rid); res["cleared"].append(rid); continue
                ent["attempts"] = int(ent.get("attempts") or 0) + 1; ent.update({"last_utc": now, "last_error": term, "last_log": log_path})
                if ent["attempts"] >= ROWS_ATTEMPT_CAP:
                    ent["state"] = ROWS_STATE_EXHAUSTED; res["exhausted"].append(rid)
                res["failed"].append(rid)
            return res
        res = rows_update(state_dir, settle)
        retried += len(res["cleared"]) + len(res["failed"]); cleared += len(res["cleared"])
        if res["stale"]:
            log_event(state_dir, {"event": "rows_retry_stale", "run_ids": res["stale"], "rc": rc, "note": "another instance settled or re-armed these items while this retry ran: left as they stand"})
        if res["cleared"]:
            log_event(state_dir, {"event": "rows_cleared", "family": first.get("family"), "run_ids": res["cleared"], "by": "retry", "attempt": attempt_n, "log": log_path,
                                  "note": "the rows step succeeded from the retained snapshot: the dispatch is complete"})
        if res["failed"]:
            log_event(state_dir, {"event": "rows_retry_failed", "run_ids": res["failed"], "rc": rc, "attempt": attempt_n, "cap": ROWS_ATTEMPT_CAP, "log": log_path,
                                  "terminal_exception": term, "stderr_tail": tail})
        if res["exhausted"]:
            log_event(state_dir, {"event": "rows_exhausted", "run_ids": sorted(res["exhausted"]), "attempts": ROWS_ATTEMPT_CAP, "snapshot_dir": key, "ledger": _rows_path(state_dir),
                                  "terminal_exception": term,
                                  "note": f"the rows step failed {ROWS_ATTEMPT_CAP} times from the retained snapshot: these runs' judgments are filed but their derived rows "
                                          f"are missing; the item stays in the ledger (counted every tick) until `{DISPATCH_ROWS_RELEASE}` names the run (or `*`)"})
    summary["rows_retried"] = int(summary.get("rows_retried", 0)) + retried
    summary["rows_cleared"] = int(summary.get("rows_cleared", 0)) + cleared


def _hold_stop(state_dir: Optional[str], summary: Dict[str, object], reason: str, stopped, where: str) -> None:
    """A hold that appeared DURING the tick -- the groups not yet dispatched are stopped (already-running work finishes; the
    hold governs NEW dispatches only), named in one `held` event with where the hold was seen."""
    summary["held"] = True; summary["hold"] = "operator"; summary["held_reason"] = reason      # the hold's KIND beside the flag
    log_event(state_dir, {"event": "held", "mid_tick": True, "hold": "operator", "reason": reason, "where": where,
                          "stopped_groups": [{"family": k[0], "model_key": k[1], "run_ids": sorted(str(r["run_id"]) for r in g)} for k, g in stopped],
                          "pending": sum(len(g) for _k, g in stopped), "file": os.path.join(str(state_dir), DISPATCH_HOLD),
                          "note": "operator dispatch hold appeared during this tick: the groups not yet dispatched are stopped, nothing more is dispatched "
                                  "or emitted this tick; already-running work finishes; cleared by removing the file"})


def dispatch_hold(state_dir: Optional[str]) -> Optional[str]:
    """The operator's hold reason (the first non-comment line of `dispatch.hold`, or "(no reason given)"), None when there is no hold."""
    if not state_dir:
        return None
    try:
        with open(os.path.join(state_dir, DISPATCH_HOLD), encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    except OSError:
        return None
    return lines[0] if lines else "(no reason given)"


#: The operator's PER-FAMILY hold -- `<state-dir>/dispatch.hold.<family>` (`dispatch.hold.petri`; the first
#: non-comment line is the reason). While the file exists every dispatch group of that family is SKIPPED by the tick: not mirrored, not
#: dispatched, not retried by rows_retry_pass, not counted as failed; its runs stay pending / quarantined exactly as they are, and a run of
#: the family that a quarantine release frees (a daemon-sha advance releases EVERY held run) is simply not dispatched that tick -- never
#: re-quarantined. Unlike the global `dispatch.hold` the family hold survives sha changes and quarantine releases: the 7 quarantined s3r
#: Petri runs must not re-dispatch (14 harvester cycles under the write lock each, until pending.claim/release's fold is fixed) when the
#: daemon advances to a new sha for other families' sake. About DISPATCH GROUPS only: the Petri DRIVE thread (the standup / drive, a
#: different thing) is covered by the global `dispatch.hold`, never by this file.
DISPATCH_HOLD_FAMILY_PREFIX = DISPATCH_HOLD + "."
_FAMILY_HOLD_SAID: Dict[str, str] = {}     # family -> the reason said (one `held_family` per (family, reason); `held_family_released` once)


def family_holds(state_dir: Optional[str]) -> Dict[str, str]:
    """`{family: reason}` for every `<state-dir>/dispatch.hold.<family>` file (the first non-comment line, or "(no reason given)"); empty
    with no state dir. The bare `dispatch.hold` (the global hold) is not a family."""
    if not state_dir:
        return {}
    try:
        names = os.listdir(state_dir)
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for name in sorted(names):
        if not name.startswith(DISPATCH_HOLD_FAMILY_PREFIX):
            continue
        fam = name[len(DISPATCH_HOLD_FAMILY_PREFIX):]
        if not fam or fam.endswith(".tmp"):
            continue
        try:
            with open(os.path.join(state_dir, name), encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
        except OSError:
            continue
        out[fam] = lines[0] if lines else "(no reason given)"
    return out


def family_hold_write(state_dir: Optional[str], family: str, reason: str, *, writer: str) -> Tuple[Optional[str], bool]:
    """Write `<state-dir>/dispatch.hold.<family>` with `reason` as its first non-comment line -- the daemon's ONE hold writer (a credential
    rejection holds the family). `(path, written)`: a hold file that already stands (the operator's, or an earlier event's) is NEVER overwritten
    and reads `written=False`; written atomically (`.tmp` + replace: `family_holds` skips `.tmp` names). No state dir: `(None, False)`."""
    if not state_dir:
        return None, False
    path = os.path.join(state_dir, DISPATCH_HOLD_FAMILY_PREFIX + family)
    if os.path.exists(path):
        return path, False
    os.makedirs(state_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(f"# written by the grading daemon ({writer}) at {utc_now()}; the operator's family hold: remove this file to lift it\n")
        fh.write(reason.replace("\n", " ").strip() + "\n")
    os.replace(tmp, path)
    return path, True


def family_holds_report(state_dir: Optional[str], holds: Mapping[str, str], counts: Mapping[str, int], where: Optional[str] = None) -> None:
    """Say each family hold ONCE per (family, reason) -- `held_family`, naming the file, the family and the runs held this tick -- re-said
    when the reason changes; and each release once -- `held_family_released` -- when the file is gone. `holds` is the COMPLETE current map
    (`family_holds(state_dir)`), never one family's slice: a family missing from it reads as RELEASED (a per-group re-read that passes a
    one-family slice announces the start-of-tick family released while its file still stands). `where` names the gate a hold was first
    seen at mid-tick (`mid_tick` on the event)."""
    for fam, reason in sorted(holds.items()):
        if _FAMILY_HOLD_SAID.get(fam) == reason:
            continue
        _FAMILY_HOLD_SAID[fam] = reason
        log_event(state_dir, {"event": "held_family", "hold": "family", "family": fam, "reason": reason, "n_runs": int(counts.get(fam, 0)),
                              "mid_tick": where is not None, "where": where,
                              "file": os.path.join(str(state_dir), DISPATCH_HOLD_FAMILY_PREFIX + fam),
                              "note": "operator family hold: every dispatch group of this family is skipped each tick -- not mirrored, not dispatched, not "
                                      "retried, not counted as failed; its runs stay pending / quarantined as they are, and a quarantine release of one "
                                      "re-dispatches nothing (never re-quarantined). Survives daemon sha changes. Dispatch groups only: the Petri DRIVE "
                                      "thread (the standup / drive) is a different thing, covered by the global dispatch.hold. Cleared by removing the file"})
    for fam in sorted(set(_FAMILY_HOLD_SAID) - set(holds)):
        was = _FAMILY_HOLD_SAID.pop(fam)
        log_event(state_dir, {"event": "held_family_released", "hold": "family", "family": fam, "was_reason": was,
                              "file": os.path.join(str(state_dir), DISPATCH_HOLD_FAMILY_PREFIX + fam),
                              "note": "the family's hold file is gone: its dispatch groups are mirrored and dispatched again from this tick on"})


def schema_gap(registry_root: str) -> Optional[Dict[str, object]]:
    """None when the live registry's migration version AND shape are the code's -- the writers' own check (`db.require_shape`:
    `migrations.db_version` against `migrations.code_version`, then `db.schema_drift` object by object) over a READ-ONLY connection
    (`db.reading`), so the preflight itself never migrates, creates or writes; else the gap: `kind` (`version` / `shape` / `check_failed`),
    `have`, `want`, `n_drift`, `drift_lines` (the first SCHEMA_HOLD_DRIFT_SHOWN), `message`. A database that cannot be read at all (absent
    mid-swap, locked past the busy timeout, not a database) is a gap of kind `check_failed`: the daemon cannot establish the schema it would
    work against, so it works against none (fail closed, never a guess, never an exception out of the loop)."""
    path = DB.db_path(registry_root)
    try:
        with DB.reading(path) as conn:
            DB.require_shape(conn, path)
    except DB.SchemaDrift as exc:
        # `fold_order`: the database records a same-stamp fold order that is not the code's (db.FoldOrderDrift; `require_shape` checks it
        # without stamping, so this read-only preflight never writes the bootstrap stamp -- a database recording no order is no gap)
        kind = "fold_order" if isinstance(exc, DB.FoldOrderDrift) else ("version" if exc.have != exc.want else "shape")
        return {"kind": kind, "have": exc.have, "want": exc.want, "n_drift": len(exc.lines),
                "drift_lines": list(exc.lines[:SCHEMA_HOLD_DRIFT_SHOWN]), "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- unreadable: a hold, never work against an unverified schema
        try:
            want: Optional[int] = RM.code_version()
        except Exception:  # noqa: BLE001
            want = None
        return {"kind": "check_failed", "have": None, "want": want, "n_drift": 0, "drift_lines": [], "message": f"{type(exc).__name__}: {exc}"}
    return None


def schema_hold_reason(gap: Mapping[str, object]) -> str:
    """One line for the tick summary's `held_reason`, the `problems` entry and the Petri event."""
    kind, have, want = gap.get("kind"), gap.get("have"), gap.get("want")
    shown = list(gap.get("drift_lines") or ())
    n = int(gap.get("n_drift") or 0)
    if kind == "version":
        return f"registry schema version {have} is not the code's {want}: run `registry migrate` (all work held until the live schema is the code's)"
    if kind == "shape":
        return (f"registry schema version {have} is the code's but its shape differs in {n} object(s) ({'; '.join(str(x) for x in shown)}"
                f"{'; …' if n > len(shown) else ''}): a migration is missing (`registry migrate --plan --sql` renders it; all work held until the "
                "live shape is the code's)")
    if kind == "fold_order":
        return (f"the registry records a fold order that is not this code's ({'; '.join(str(x) for x in shown)}): run the code the database "
                "was folded by, or refold it under this code's order (`registry rebuild`); all work held until they agree")
    return f"the registry's schema could not be read ({gap.get('message')}): no work against an unverified schema"


class SchemaGate:
    """The schema hold's state, shared by the grading loop and the Petri thread. `check` re-reads the live schema every time it is
    called (the top of every tick, every Petri start decision, once in main() before either loop starts) and says the TRANSITIONS once: one
    `schema_hold` event per distinct gap (the same gap on later checks is silent; a gap that changes -- a further drift line, a version that
    moved -- is a new event) and one `schema_hold_released` when the live schema reaches the code's version and shape. Nothing here migrates:
    the release is observed, never performed. Thread-safe (the two loops check concurrently)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: Optional[Dict[str, object]] = None
        self._said: Optional[tuple] = None

    @property
    def held(self) -> Optional[Dict[str, object]]:
        with self._lock:
            return dict(self._held) if self._held is not None else None

    def check(self, registry_root: Optional[str], state_dir: Optional[str]) -> Optional[Dict[str, object]]:
        """The current gap (a copy, `since_utc` added) or None; logs the transition events. No registry named: no check, no hold."""
        if not registry_root:
            return None
        gap = schema_gap(registry_root)
        now = utc_now()
        with self._lock:
            if gap is None:
                if self._held is not None:
                    was = self._held
                    try:
                        version: Optional[int] = RM.code_version()
                    except Exception:  # noqa: BLE001
                        version = was.get("want")  # type: ignore[assignment]
                    log_event(state_dir, {"event": "schema_hold_released", "have": version, "want": version, "was_have": was.get("have"), "was_kind": was.get("kind"),
                                          "held_since_utc": was.get("since_utc"),
                                          "note": "the live registry's schema is the code's version and shape again: grading, the Petri start and the contrast "
                                                  "passes resume from this check on (the daemon migrated nothing; the hold only observed it)"})
                self._held = None
                self._said = None
                return None
            key = (gap["kind"], gap["have"], gap["want"], gap["n_drift"], tuple(gap["drift_lines"]))
            gap["since_utc"] = self._held["since_utc"] if self._held is not None else now
            if key != self._said:
                log_event(state_dir, {"event": "schema_hold", **gap, "reason": schema_hold_reason(gap),
                                      "note": "the live registry's migration version or shape is not this daemon's code's: NO work -- no rows retry, claim "
                                              "reconciliation, pending read, mirror, dispatch, Petri start or contrast -- until it is (re-checked every tick and "
                                              "at every Petri start decision; released by itself when `registry migrate` completes; the daemon keeps ticking)"})
                self._said = key
            self._held = gap
            return dict(gap)


SCHEMA_GATE = SchemaGate()


def _schema_stop(state_dir: Optional[str], summary: Dict[str, object], gap: Mapping[str, object], stopped, where: str) -> None:
    """The schema gap seen DURING the tick (a grader's claim refused it) -- the groups not yet dispatched are stopped, named in one `held`
    event of kind `schema`; the summary says held and why (the next tick's preflight takes over)."""
    summary["held"] = True; summary["hold"] = "schema"; summary["held_reason"] = schema_hold_reason(gap)
    summary["schema"] = {k: gap.get(k) for k in ("kind", "have", "want", "n_drift")}
    log_event(state_dir, {"event": "held", "mid_tick": True, "hold": "schema", "reason": summary["held_reason"], "where": where,
                          "stopped_groups": [{"family": k[0], "model_key": k[1], "run_ids": sorted(str(r["run_id"]) for r in g)} for k, g in stopped],
                          "pending": sum(len(g) for _k, g in stopped),
                          "note": "the registry's schema is not the code's any more: the groups not yet dispatched are stopped, nothing more is dispatched, "
                                  "emitted or pruned this tick; the next tick's preflight holds until the live schema is the code's"})


def petri_start_hold(state_dir: Optional[str], registry_root: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """`(kind, reason)` of the hold that forbids a Petri drive START right now -- the operator's `dispatch.hold` (`operator`) or the schema
    gap (`schema`, through SCHEMA_GATE, so the transition events are said once across both loops); `(None, None)` when a start may go ahead."""
    reason = dispatch_hold(state_dir)
    if reason is not None:
        return "operator", reason
    gap = SCHEMA_GATE.check(registry_root, state_dir)
    if gap is not None:
        return "schema", schema_hold_reason(gap)
    return None, None


def _release_requests(state_dir: str, consume: bool) -> Optional[set]:
    """The explicit release file's run ids ({'*'} for all), consumed when read; None when there is no file."""
    path = os.path.join(state_dir, DISPATCH_QUARANTINE_RELEASE)
    try:
        with open(path, encoding="utf-8") as fh:
            ids = {ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")}
    except OSError:
        return None
    if consume:
        try:
            os.unlink(path)
        except OSError:
            pass
    return ids


def quarantine_filter(state_dir: Optional[str], registry_root: str, rows: Sequence[Dict[str, object]], *, dry_run: bool = False
                      ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Release every quarantined run whose cause changed (daemon sha, store generation, the release file -- and a VERIFIED
    change of the run's identity binding: the record's `identity_binding` {instrument_id, required_judge_id} at filing time differs from what the
    pending view states for the run now, so the quarantined judgment was made against an identity since corrected -- never a blanket release on
    a pending_reason, never for a record that carries no binding), drop the ones no longer pending, then split `rows` into (eligible, held). A
    dry run reads but neither saves nor consumes the release file.

    The release file is read, consumed and VALIDATED every tick, an EMPTY ledger included -- an empty ledger that returned before
    `_release_requests` ran let a file written while nothing was held survive every tick and release the NEXT failure on sight. A
    requested id not presently in the ledger is rejected with one `quarantine_release_rejected` event (never stored, never applied
    later); a file found with an empty ledger is consumed and logged once (`quarantine_release_ignored_empty`; `*` is a no-op there).
    `*` keeps its meaning for the ids that ARE held."""
    runs = quarantine_load(state_dir)
    if not state_dir:
        return list(rows), []
    sha = IMPORTED_CODE_SHA          # the running code: a quarantine stamped by an older import is released for the fix it never saw
    asked = _release_requests(state_dir, consume=not dry_run)     # read + consumed (not on a dry run) BEFORE the empty-ledger return
    if asked is not None:
        for rid in sorted(a for a in asked if a != "*" and a not in runs):
            log_event(state_dir, {"event": "quarantine_release_rejected", "run_id": rid, "reason": "not quarantined", "dry_run": dry_run,
                                  "ledger": _quarantine_path(state_dir), "n_quarantined": len(runs),
                                  "note": f"`{DISPATCH_QUARANTINE_RELEASE}` names a run the ledger does not hold: rejected, not stored, never applied "
                                          "to a later quarantine of that run (D2610 H1); the file is consumed with the rest of its lines"})
        if not runs:
            log_event(state_dir, {"event": "quarantine_release_ignored_empty", "requested": sorted(asked), "n": len(asked), "dry_run": dry_run,
                                  "ledger": _quarantine_path(state_dir),
                                  "note": f"`{DISPATCH_QUARANTINE_RELEASE}` found with an empty ledger: consumed and ignored (`*` included) -- a release "
                                          "written while nothing is quarantined never releases a later failure (D2610 H1)"})
        asked = {a for a in asked if a == "*" or a in runs}
    else:
        asked = set()
    if not runs:
        return list(rows), []
    released: List[Tuple[str, str]] = []
    rebound: Dict[str, Tuple[Optional[Dict[str, Optional[str]]], Optional[Dict[str, Optional[str]]]]] = {}
    dropped: List[str] = []
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            for rid, ent in list(runs.items()):
                if "*" in asked or rid in asked:
                    released.append((rid, "release_file"))
                elif str(ent.get("daemon_sha") or "") != sha:
                    released.append((rid, "daemon_sha"))
                elif store_generation(conn, rid, None) != str(ent.get("store_generation") or ""):
                    released.append((rid, "store_generation"))
                else:
                    pend_rows = P.pending(conn, None, ent.get("campaign"), run_id=rid)
                    if not pend_rows:
                        dropped.append(rid)
                    else:
                        was, cur = ent.get("identity_binding"), identity_binding_of(pend_rows[0])
                        if identity_binding_differs(was, cur):                 # the identity the quarantined judgment was made against is gone
                            released.append((rid, "identity_rebound")); rebound[rid] = (dict(was) if was else None, cur)
    except Exception as exc:  # noqa: BLE001 — no registry answer: nothing released on generation, identity or settlement; the file and the sha still count
        log_event(state_dir, {"event": "quarantine_check_deferred", "error": f"{type(exc).__name__}: {exc}"})
        for rid, ent in list(runs.items()):
            if "*" in asked or rid in asked:
                released.append((rid, "release_file"))
            elif str(ent.get("daemon_sha") or "") != sha:
                released.append((rid, "daemon_sha"))
    for rid, why in released:
        ent = runs.pop(rid, {})
        ev = {"event": "quarantine_released", "run_id": rid, "reason": why, "family": ent.get("family"), "quarantined_utc": ent.get("utc"),
              "was_daemon_sha": ent.get("daemon_sha"), "daemon_sha": sha}
        if why == "identity_rebound":
            was, cur = rebound.get(rid, (ent.get("identity_binding"), None))
            ev.update({"was_binding": was, "binding": cur, "judgment_ids": ent.get("judgment_ids"), "judgment_bindings": ent.get("judgment_bindings"),
                       "note": "the run's identity binding (instrument / required judge) changed since the quarantined judgment was made: that judgment "
                               "counts for nothing under the current identity; released for one re-dispatch against it"})
        log_event(state_dir, ev)
    for rid in dropped:
        ent = runs.pop(rid, {})
        log_event(state_dir, {"event": "quarantine_dropped", "run_id": rid, "family": ent.get("family"), "note": "no longer pending judgment: settled by another route"})
    if (released or dropped) and not dry_run:
        quarantine_save(state_dir, runs)
    held = [r for r in rows if str(r["run_id"]) in runs]
    eligible = [r for r in rows if str(r["run_id"]) not in runs]
    if held:
        fams: Dict[str, int] = {}
        for r in held:
            fams[str(r.get("family"))] = fams.get(str(r.get("family")), 0) + 1
        log_event(state_dir, {"event": "problem", "kind": "dispatch_quarantine", "n": len(held), "families": fams, "run_ids": sorted(str(r["run_id"]) for r in held),
                              "ledger": _quarantine_path(state_dir),
                              "note": f"{len(held)} pending run(s) held after a failed dispatch: read the ledger's stderr_tail, fix the cause, then release "
                                      f"with a new daemon sha, a re-filed store, or `{DISPATCH_QUARANTINE_RELEASE}` in the state dir"})
    return eligible, held


def run_pending(conn, run_id: str, campaign: Optional[str]) -> bool:
    """The worklist's OWN question: is this run still in `v_pending_judgment` for its instrument's judge (`pending.pending(conn, None,
    campaign, run_id=…)`)? A coverage-short or wrong-judge complete judgment, a claim, a partial: all still pending — its store stays."""
    return bool(P.pending(conn, None, campaign, run_id=run_id))


def _dir_in_use(path: str) -> bool:
    """Some other process has `path` (or a directory under it) as its working directory — a reader that may still hold the copy."""
    me = os.getpid(); real = os.path.realpath(path)
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if cwd == real or cwd.startswith(real + os.sep):
            return True
    return False


def mirror_settle(state_dir: Optional[str], registry_root: str, *, family: str, model_key: str, campaign: Optional[str], run_ids: Sequence[str],
                  mirrored: Sequence[Mapping[str, object]], rc, reason: str = "this tick's mirror or dispatch failed", retryable: bool = False) -> Dict[str, int]:
    """After the group's dispatch RETURNED: a clean tick (every rsync rc 0, dispatch rc 0) queues its directories for the prune; a tick whose
    rsync or dispatch failed RETIRES the incomplete copy of every run the worklist still shows pending (`mirror_retired_incomplete`, with bytes —
    the dispatch it served has returned, and no other process holds the directory), so the next tick's floor check sees the space again and
    re-mirrors from scratch instead of wedging under the floor; directories of runs no longer pending are queued as usual. Returns counts."""
    out = {"retired": 0, "retired_bytes": 0, "queued": 0}
    if not mirrored:
        return out
    clean = (rc == 0 or retryable) and all(m.get("rc") == 0 for m in mirrored)   # a retryable dispatch failure keeps the complete copy for the next tick
    if clean:
        mirror_prune_enqueue(state_dir, family=family, model_key=model_key, campaign=campaign, run_ids=run_ids, mirrored=mirrored, rc=rc)
        out["queued"] = len(mirrored)
        return out
    keep: List[Mapping[str, object]] = []
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            still = {str(m["run_id"]): run_pending(conn, str(m["run_id"]), campaign) for m in mirrored}
    except Exception as exc:  # noqa: BLE001 — no registry answer: retire nothing, queue everything (the prune pass asks again later)
        log_event(state_dir, {"event": "mirror_retire_deferred", "error": f"{type(exc).__name__}: {exc}", "run_ids": list(run_ids)})
        mirror_prune_enqueue(state_dir, family=family, model_key=model_key, campaign=campaign, run_ids=run_ids, mirrored=mirrored, rc=rc)
        return dict(out, queued=len(mirrored))
    # A copy an OPEN derived-rows item references -- by the run id, or as one of the item's recorded mirrored
    # store directories; any state, `awaiting_grader` included -- is never retired: the judgments it holds are spooled (the run still reads
    # pending until the fold) and the rows retry reads its judged.jsonl from exactly this directory. Kept and queued for the prune pass, which
    # defers the group while the item is open.
    rows_open = rows_load(state_dir); rows_open_runs = set(rows_open)
    rows_open_dirs = {os.path.realpath(str(d)) for e in rows_open.values() for d in e.get("dirs") or [] if d}
    for m in mirrored:
        local, root = str(m.get("local") or ""), str(m.get("root") or "")
        if not still.get(str(m["run_id"])) or not root or not _under_root(root, local) or not os.path.isdir(local):
            keep.append(m); continue
        if str(m["run_id"]) in rows_open_runs or os.path.realpath(local) in rows_open_dirs:
            log_event(state_dir, {"event": "mirror_retire_deferred", "run_id": m["run_id"], "dir": local, "reason": "derived rows pending",
                                  "note": "an open derived-rows item still reads this judged store (g29-fix-5): kept and queued, never retired under it"})
            keep.append(m); continue
        if _dir_in_use(local):
            log_event(state_dir, {"event": "mirror_retire_deferred", "run_id": m["run_id"], "dir": local, "note": "another process holds the directory; left for the next tick's rsync to resume"})
            continue
        size = _du(local)
        shutil.rmtree(local, ignore_errors=True)
        out["retired"] += 1; out["retired_bytes"] += size
        log_event(state_dir, {"event": "mirror_retired_incomplete", "run_id": m["run_id"], "dir": local, "bytes": size, "rsync_rc": m.get("rc"), "dispatch_rc": rc,
                              "note": f"the run is still pending and {reason}: the incomplete copy is gone; the next tick re-mirrors it"})
    if keep:
        mirror_prune_enqueue(state_dir, family=family, model_key=model_key, campaign=campaign, run_ids=run_ids, mirrored=keep, rc=rc)
        out["queued"] = len(keep)
    return out


def mirror_prune_pass(registry_root: str, state_dir: Optional[str], dry_run: bool, campaign: Optional[str] = None) -> Dict[str, int]:
    """Remove the queued store directories of every group NONE of whose runs is still pending (the worklist's own predicate, `run_pending`):
    one `mirror_pruned` event per directory with its bytes; a group with a pending run stays queued. Returns {groups, dirs, bytes}."""
    out = {"groups": 0, "dirs": 0, "bytes": 0}
    pending = _prune_ledger_load(state_dir)
    if not pending or dry_run:
        return out
    keep: List[dict] = []
    rows_open = set(rows_load(state_dir))                   # a run with an open derived-rows item still needs its judged store
    try:
        with DB.reading(DB.db_path(registry_root)) as conn:
            for entry in pending:
                camp = entry.get("campaign") if "campaign" in entry else campaign
                try:
                    done = bool(entry.get("run_ids")) and not any(run_pending(conn, rid, camp) for rid in entry.get("run_ids") or [])
                except Exception as exc:  # noqa: BLE001
                    log_event(state_dir, {"event": "mirror_prune_deferred", "error": f"{type(exc).__name__}: {exc}", "run_ids": entry.get("run_ids")})
                    keep.append(entry); continue
                if not done:
                    keep.append(entry); continue
                if rows_open and {str(x) for x in entry.get("run_ids") or []} & rows_open:
                    log_event(state_dir, {"event": "mirror_prune_deferred", "reason": "derived rows pending", "run_ids": entry.get("run_ids"),
                                          "note": "the rows retry reads these judged stores from the mirror: kept until the rows item clears (g29-fix-4)"})
                    keep.append(entry); continue
                with mirror_locks([str(d.get("local") or "") for d in entry.get("dirs") or [] if d.get("local")]) as held:
                    if held.busy:   # another instance (or an orphaned grader) holds one of them: this group waits for a later tick
                        log_event(state_dir, {"event": "mirror_prune_deferred", "reason": "locked by another instance", "dirs": held.busy, "run_ids": entry.get("run_ids")})
                        keep.append(entry); continue
                    for d in entry.get("dirs") or []:
                        local, root = str(d.get("local") or ""), str(d.get("root") or "")
                        if not root or not _under_root(root, local) or not os.path.isdir(local):
                            continue
                        size = _du(local)
                        shutil.rmtree(local, ignore_errors=True)
                        out["dirs"] += 1; out["bytes"] += size
                        log_event(state_dir, {"event": "mirror_pruned", "run_id": d.get("run_id"), "dir": local, "bytes": size, "family": entry.get("family"), "model_key": entry.get("model_key")})
                    out["groups"] += 1
    except Exception as exc:  # noqa: BLE001 — no registry, no confirmation: nothing is pruned
        log_event(state_dir, {"event": "mirror_prune_deferred", "error": f"{type(exc).__name__}: {exc}"})
        return out
    _prune_ledger_save(state_dir, keep)  # type: ignore[arg-type]
    return out


def emit_contrasts_for(registry_root: str, model_ids: Sequence[str], campaign: Optional[str], spool: Optional[str], dry_run: bool,
                       state_dir: Optional[str]) -> int:
    """`scripts/registry/emit_contrasts.py` (beside this file): one `run_contrast` per steered run whose anchor is complete. A failing
    pass is one `contrasts_failed` event and never stops the grading loop (the drivers' rows are unaffected)."""
    try:
        n = EC.run(registry_root=registry_root, model_ids=list(model_ids), campaign=campaign, spool=spool, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 — a contrast pass never stops the grading loop
        log_event(state_dir, {"event": "contrasts_failed", "error": f"{type(exc).__name__}: {exc}"})
        return 0
    log_event(state_dir, {"event": "contrasts", "written": n, "dry_run": dry_run})
    return int(n or 0)


def contrasts_sweep_hold(state_dir: Optional[str]) -> Optional[str]:
    """sweephold: the reason on the sweep hold file (`<state-dir>/contrasts_sweep_hold`: its first non-comment line, or "(no reason
    given)" for an empty file), None when there is no hold or no state dir. Read every time it is asked -- never cached -- so a flip of the
    file takes effect at the next tick."""
    if not state_dir:
        return None
    try:
        with open(os.path.join(state_dir, CONTRASTS_SWEEP_HOLD), encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    except OSError:
        return None
    return lines[0] if lines else "(no reason given)"


def contrasts_sweep_hold_seeding(state_dir: Optional[str]) -> Optional[Dict[str, object]]:
    """sweephold-fix-1: the persisted record of the ONE seeding this state dir ever had (`<state-dir>/contrasts_sweep_hold.seeded`: utc,
    daemon_sha, pid, wrote_file, reason), None when no flagged start has applied its initial state here (or no state dir)."""
    if not state_dir:
        return None
    try:
        with open(os.path.join(state_dir, CONTRASTS_SWEEP_HOLD_SEEDED), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else {"raw": rec}


def _sweephold_event_is_dry_run(ev: Mapping[str, object]) -> bool:
    """sweephold-fix-3: an event a dry-run daemon wrote -- stamped `dry_run: true` (every sweephold-era event since fix-3), or a fix-1 / fix-2
    dry-run start-up, whose seeding dict said `skipped: dry_run` before the stamp existed."""
    if ev.get("dry_run"):
        return True
    seeding = ev.get("contrasts_sweep_hold_seeding")
    return isinstance(seeding, dict) and seeding.get("skipped") == "dry_run"


def _sweephold_event_class(ev: Mapping[str, object]) -> Optional[str]:
    """sweephold-fix-5: the class an event states for ITSELF, by an affirmative discriminator only -- SWEEPHOLD_CLASS_DRY_RUN (stamped `dry_run:
    true`, a seeding dict saying `skipped: dry_run`, a tick whose `contrasts_sweep_hold.sweep` reads held_by_flag_dry_run), SWEEPHOLD_CLASS_REAL
    (stamped `dry_run: false` -- every event since fix-3 -- or a `daemon_started` whose seeding WROTE a file: `contrasts_sweep_hold_seeded` set, or a
    seeding dict with `seeded_now` set or `seeding_recorded`; a dry run never writes one -- or a seeding dict saying `seeded_before` or `skipped:
    ambiguous_legacy_state`, returns that lie PAST the dry-run return at every unstamped version that wrote a dict: fix-1 00cf8b96c and fix-2
    dd138fb9d both order no_flag -> no_state_dir -> dry_run -> seeded_before -> ambiguous, so such a start was not a dry run by construction),
    else None: an unstamped event that says nothing about itself (f03e6ee76's `contrasts_sweep_hold_seeded: null` with no seeding dict,
    00cf8b96c's `skipped: no_flag` -- the returns BEFORE the dry-run check -- and the unstamped ticks of either) -- never real by default."""
    if _sweephold_event_is_dry_run(ev):
        return SWEEPHOLD_CLASS_DRY_RUN
    csh = ev.get("contrasts_sweep_hold")
    if isinstance(csh, dict) and csh.get("sweep") == SWEEP_STATUS_HELD_DRY_FLAG:
        return SWEEPHOLD_CLASS_DRY_RUN
    if ev.get("dry_run") is False:
        return SWEEPHOLD_CLASS_REAL
    if ev.get("event") == "daemon_started":
        seeding = ev.get("contrasts_sweep_hold_seeding")
        if ev.get("contrasts_sweep_hold_seeded") or (isinstance(seeding, dict) and (seeding.get("seeded_now") or seeding.get("seeding_recorded"))):
            return SWEEPHOLD_CLASS_REAL                 # a file was written: a dry run never writes one
        if isinstance(seeding, dict) and (seeding.get("seeded_before") or seeding.get("skipped") == SWEEPHOLD_SKIP_AMBIGUOUS):
            return SWEEPHOLD_CLASS_REAL                 # past the dry-run return at every unstamped version that wrote a seeding dict (fix-1 00cf8b96c,
                                                        # fix-2 dd138fb9d: no_flag -> no_state_dir -> dry_run -> seeded_before -> ambiguous): real by construction
    return None


#: sweephold-fix-4: the three classes of an era event in the state dir's log (`sweephold_classify`)
SWEEPHOLD_CLASS_DRY_RUN = "dry_run"; SWEEPHOLD_CLASS_REAL = "real"; SWEEPHOLD_CLASS_UNCLASSIFIED = "unclassified"


def _sweephold_log_chunks(state_dir: str) -> List[str]:
    """The scanned text of the state dir's event log: its first and last SWEEPHOLD_EVIDENCE_SCAN_BYTES (the whole file when smaller), decoded;
    [] when there is no log."""
    log = os.path.join(state_dir, "grade_daemon.jsonl")
    try:
        size = os.path.getsize(log)
    except OSError:
        return []
    chunks: List[bytes] = []
    try:
        with open(log, "rb") as fh:
            chunks.append(fh.read(SWEEPHOLD_EVIDENCE_SCAN_BYTES))
            if size > SWEEPHOLD_EVIDENCE_SCAN_BYTES:
                fh.seek(max(SWEEPHOLD_EVIDENCE_SCAN_BYTES, size - SWEEPHOLD_EVIDENCE_SCAN_BYTES)); chunks.append(fh.read())
    except OSError:
        return []
    return [c.decode("utf-8", errors="replace") for c in chunks]


def sweephold_classify(state_dir: Optional[str]) -> List[Dict[str, object]]:
    """Every sweephold-era event in the scanned log, in log order, with the CLASS its run context gives it -- ``{event, utc, class, window}``
    where class is `dry_run` (the event states it itself -- `_sweephold_event_class` -- or it sits in a dry-run start-up's window), `real`
    (it states it itself, or it sits in a real start-up's window) or `unclassified` (it states nothing and sits in the window of a start-up
    that states nothing either -- an old start without a `dry_run` stamp, a `skipped: dry_run` or a seeding that wrote a file is NEVER real
    by default -- or no start-up window places it: the start-up outside the scanned text, or a `daemon_exit` closed the window and no start
    followed). A window opens at every `daemon_started` (the daemon's re-exec on code drift keeps its pid, so one pid opens several) with the
    class that start states for itself, and closes at the next `daemon_started` or at a `daemon_exit`; an event that carries a `pid` belongs
    to the window only when it is the start-up's pid (ticks record none). The classes are what `sweephold_history` counts: a dry-run audit's
    older unstamped ticks are noise, never evidence of a real sweephold era, and an unclassifiable start-up or tick is reported, never
    counted and never silently seeded over."""
    if not state_dir:
        return []
    out: List[Dict[str, object]] = []
    window: Optional[Dict[str, object]] = None          # the open start-up: {pid, cls, utc}; None = no start-up placed yet / closed by an exit
    for text in _sweephold_log_chunks(state_dir):
        window = None                                   # the two scanned chunks are not contiguous: a window never spans them
        for raw in text.splitlines():
            if "contrasts_sweep_hold" not in raw and '"daemon_started"' not in raw and '"daemon_exit"' not in raw:
                continue
            try:
                ev = json.loads(raw)
            except ValueError:
                continue                                   # a line cut by the scan window, or not JSON
            if not isinstance(ev, dict):
                continue
            kind = ev.get("event")
            if kind == "daemon_started":
                window = {"pid": ev.get("pid"), "cls": _sweephold_event_class(ev) or SWEEPHOLD_CLASS_UNCLASSIFIED, "utc": str(ev.get("utc") or "")}
                if any(k in ev for k in SWEEPHOLD_SEEDING_EVENT_KEYS):
                    out.append({"event": kind, "utc": window["utc"], "class": window["cls"], "window": window["utc"]})
                continue
            if kind == "daemon_exit":
                if window is None or ev.get("pid") is None or window.get("pid") is None or ev.get("pid") == window.get("pid"):
                    window = None                              # this start-up ended: later events have no window until the next start
                continue
            if not (kind == "contrasts_sweep_held" or (kind == "tick" and "contrasts_sweep_hold" in ev)):
                continue
            own = _sweephold_event_class(ev)
            placed = window if (window is not None and (ev.get("pid") is None or window.get("pid") is None or ev.get("pid") == window.get("pid"))) else None
            if own is not None:
                cls = own                                       # (a) the event says so itself (a stamp, a dry-run status)
            elif placed is None:
                cls = SWEEPHOLD_CLASS_UNCLASSIFIED              # (d) no start-up window places it
            else:
                cls = str(placed["cls"])                        # (b) / (c) the start-up's own class -- dry_run, real, or (fix-5) unclassified
            out.append({"event": kind, "utc": str(ev.get("utc") or ""), "class": cls, "window": placed["utc"] if placed is not None else None})
    return out


def sweephold_history(state_dir: Optional[str]) -> Dict[str, object]:
    """sweephold-fix-2 / fix-3 / fix-4: what this state dir says about earlier daemons on sweephold code -- ``{"evidence": [one line per kind
    found], "dry_run_events_skipped": n[, "unclassified": {ticks, held_events, first_utc, last_utc}]}``. Evidence is read from the state dir's
    own event log `grade_daemon.jsonl` (its first and last SWEEPHOLD_EVIDENCE_SCAN_BYTES: a REAL `daemon_started` carrying
    `contrasts_sweep_hold_seeded` / `contrasts_sweep_hold_seeding`, a REAL `contrasts_sweep_held` event, a REAL `tick` carrying
    `contrasts_sweep_hold`) and from the mark `contrasts_sweep.json` (`sweephold_era`, written by every mark since fix-2; a dry run writes no
    mark). fix-4: REAL means classified so by its run context (:func:`sweephold_classify`) -- a dry-run daemon's events, stamped or merely
    inside a dry-run start-up's window, are never evidence (an audit cannot alter the live daemon's initial hold) and are counted under
    `dry_run_events_skipped`; an event no start-up window places is reported under `unclassified` (present only when there is one) and
    never counted. The hold file's presence is not evidence."""
    if not state_dir:
        return {"evidence": [], "dry_run_events_skipped": 0}
    out: List[str] = []
    noise = 0
    started = held = ticks = 0; first_utc: Optional[str] = None
    unc_ticks = unc_held = unc_starts = 0; unc_first: Optional[str] = None; unc_last: Optional[str] = None
    for c in sweephold_classify(state_dir):
        kind, cls = str(c["event"]), str(c["class"])
        if cls == SWEEPHOLD_CLASS_DRY_RUN:
            noise += 1
            continue
        if cls == SWEEPHOLD_CLASS_UNCLASSIFIED:
            if kind == "tick":
                unc_ticks += 1
            elif kind == "daemon_started":
                unc_starts += 1                                # fix-5: an old start-up without an affirmative discriminator
            else:
                unc_held += 1
            unc_first = unc_first or str(c["utc"]); unc_last = str(c["utc"])
            continue
        if kind == "daemon_started":
            started += 1; first_utc = first_utc or str(c["utc"])
        elif kind == "contrasts_sweep_held":
            held += 1
        else:
            ticks += 1
    if started:
        out.append(f"event log: {started} daemon_started event(s) carrying contrasts_sweep_hold_seeded / contrasts_sweep_hold_seeding (first at {first_utc})")
    if held:
        out.append(f"event log: {held} contrasts_sweep_held event(s)")
    if ticks:
        out.append(f"event log: {ticks} tick event(s) carrying contrasts_sweep_hold")
    try:
        with open(os.path.join(state_dir, CONTRASTS_SWEEP_MARK), encoding="utf-8") as fh:
            mark = json.load(fh)
    except (OSError, ValueError):
        mark = None
    if isinstance(mark, dict) and mark.get("sweephold_era"):
        out.append(f"mark: {CONTRASTS_SWEEP_MARK} carries sweephold_era (daemon_sha {mark.get('daemon_sha')}, {mark.get('utc')})")
    hist: Dict[str, object] = {"evidence": out, "dry_run_events_skipped": noise}
    if unc_ticks or unc_held or unc_starts:
        hist["unclassified"] = {"starts": unc_starts, "ticks": unc_ticks, "held_events": unc_held, "first_utc": unc_first, "last_utc": unc_last,
                                "note": "sweephold-era events nobody can classify: an old start-up without an affirmative discriminator (a dry_run stamp, or a "
                                        "seeding that wrote a file) and the ticks in its window (sweephold-fix-5), or events no start-up window in the scanned "
                                        "log places (the start-up outside the scanned text, or after a daemon_exit): reported, not counted as evidence, never "
                                        "silently seeded over; a flagged start holds the sweep until the operator decides"}
    return hist


def sweephold_evidence(state_dir: Optional[str]) -> List[str]:
    """The evidence lines of :func:`sweephold_history` alone ([] for none; dry-run events excluded)."""
    return list(sweephold_history(state_dir)["evidence"])  # type: ignore[arg-type]


def contrasts_sweep_pending_decision(state_dir: Optional[str]) -> Optional[Dict[str, object]]:
    """sweephold-fix-6: the persisted UNRESOLVED pending decision -- `<state-dir>/contrasts_sweep_pending_decision`, written by the first
    derivation of `held_by_flag_pending_decision` on this state dir (`_sweephold_pending_write`) -- or None when there is none (or no state dir).
    A record that exists but cannot be read still HOLDS: it comes back as ``{"unreadable": "<error>"}``, never as absent (the hold fails toward
    holding; the operator's two paths clear it either way)."""
    if not state_dir:
        return None
    path = os.path.join(state_dir, CONTRASTS_SWEEP_PENDING_DECISION)
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        return {"unreadable": f"{type(exc).__name__}: {exc}"}
    return rec if isinstance(rec, dict) else {"raw": rec}


def _sweephold_pending_unclassified(rec: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """The record's unclassified block ({} for an unreadable or malformed record)."""
    unc = rec.get("unclassified") if isinstance(rec, Mapping) else None
    return dict(unc) if isinstance(unc, Mapping) else {}


def _sweephold_pending_standing(rec: Optional[Mapping[str, object]], written_now: bool) -> str:
    """How the record stands, for the operator's reading: written by this derivation, standing since an earlier one, or unreadable."""
    if written_now:
        return "written now"
    if isinstance(rec, Mapping) and rec.get("unreadable"):
        return f"present but unreadable: {rec.get('unreadable')}"
    utc = rec.get("utc") if isinstance(rec, Mapping) else None; sha = rec.get("daemon_sha") if isinstance(rec, Mapping) else None
    return f"standing: derived {utc} by daemon {sha}"


def _sweephold_pending_write(state_dir: str, history: Mapping[str, object]) -> Tuple[Optional[Dict[str, object]], bool]:
    """sweephold-fix-6: PERSIST the pending decision ``history`` (a `sweephold_history` reading with an `unclassified` block) derives -- the record
    `<state-dir>/contrasts_sweep_pending_decision`, written ONCE (the first derivation, dated; later starts and ticks read it and never rewrite it):
    the unclassified counts and first / last utc exactly as the history reported them, the real evidence lines beside them, the deriving daemon's
    sha / pid / utc, and the two operator paths as text. One `sweep_pending_decision_recorded` event (stamped `dry_run: false` -- a dry run never
    reaches this). sweephold-fix-7: the publish is RACE-FREE. fix-6 wrote through ONE fixed `.tmp`
    path and an unconditional os.replace, so two real flagged starts on one state dir raced -- the loser died at start-up with FileNotFoundError (its
    tmp renamed away by the winner) before `daemon_started`, or both published (a double record, two "one-time" events). Now the body goes to a
    UNIQUE tmp (tempfile.mkstemp in the state dir, complete before anything can see it) and is published EXCLUSIVELY with os.link, which fails EEXIST
    when a winner's record exists; the loser reads the winner's record back and proceeds held on it, saying nothing of its own (the winner said the
    event); the tmp is removed either way. Returns ``(record, written)``: ``(this record, True)`` when this process published; ``(the winner's record,
    False)`` when another did first -- the callers report `written_now: false` with it; ``(None, False)`` in the degenerate case only: the record it
    collided with vanished by the read-back (the operator's `.seeded` honoured that instant -- then nothing is pending -- or an rm, retried
    SWEEPHOLD_PENDING_PUBLISH_ATTEMPTS times): nothing is persisted and nothing pending is reported; the next tick derives again. Never raises for
    losing the race."""
    unc = dict(history.get("unclassified") or {})  # type: ignore[arg-type]
    unc.pop("note", None)
    hold = os.path.join(state_dir, CONTRASTS_SWEEP_HOLD); seeded = os.path.join(state_dir, CONTRASTS_SWEEP_HOLD_SEEDED)
    path = os.path.join(state_dir, CONTRASTS_SWEEP_PENDING_DECISION)
    rec: Dict[str, object] = {"utc": utc_now(), "daemon_sha": IMPORTED_CODE_SHA, "pid": os.getpid(), "unclassified": unc,
                              "evidence": list(history.get("evidence") or []), "hold_file": hold, "seeded_record": seeded,  # type: ignore[arg-type]
                              "operator_decision": (f"touch {hold} to hold the sweep (this record stands, superseded while the file exists), or write {seeded} "
                                                    "(any JSON object, e.g. {\"acknowledged_by\": \"<who>\"}) to acknowledge the released state (this record is then "
                                                    "removed and the sweep runs once per advance); nothing else clears it"),
                              "note": "sweephold-fix-6 (D2206): the first derivation of held_by_flag_pending_decision on this state dir, persisted so that the "
                                      "unclassified events leaving the bounded evidence scan (the log's first and last 4 MiB) never release the sweep without an "
                                      "operator decision: a flagged, non-dry daemon holds the once-per-advance sweep and seeds nothing while this file exists, "
                                      "whatever the scan sees; an unflagged daemon is not bound by it; a dry run neither writes nor removes it"}
    os.makedirs(state_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=state_dir, prefix=f".{CONTRASTS_SWEEP_PENDING_DECISION}.", suffix=".tmp")   # fix-7: unique per process, never shared
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fd, 0o644)
            json.dump(rec, fh, indent=2, sort_keys=True); fh.write("\n")
        for _attempt in range(SWEEPHOLD_PENDING_PUBLISH_ATTEMPTS):
            try:
                os.link(tmp, path)                                 # fix-7: the EXCLUSIVE publish -- EEXIST when a winner's record is already there
            except FileExistsError:
                winner = contrasts_sweep_pending_decision(state_dir)
                if winner is not None:
                    return winner, False                           # lost the race: the winner's record binds this process too, and the winner said the event
                if contrasts_sweep_hold_seeding(state_dir) is not None:
                    return None, False                             # the record vanished because the operator's .seeded record was honoured: nothing pending
                continue                                           # vanished otherwise (an rm between the EEXIST and the read-back): publish again
            log_event(state_dir, {"event": "sweep_pending_decision_recorded", "dry_run": False, "path": path, "record": rec,
                                  "note": "the operator's decision over unreadable sweephold history is pending and now persisted: the once-per-advance sweep is held on "
                                          "every flagged start until the hold file or the .seeded record exists (sweephold-fix-6)"})
            return rec, True
        return None, False                                         # the record kept vanishing under the publish: nothing persisted this time
    finally:
        try:
            os.unlink(tmp)                                         # the winner's tmp (the published path keeps the inode) and the loser's alike
        except FileNotFoundError:
            pass


def _sweephold_pending_clear(state_dir: str) -> Optional[str]:
    """sweephold-fix-6: remove the pending record because the operator's `.seeded` record now stands -- the ONE release (the hold file supersedes the
    record without removing it; eviction, a re-exec, a clean-looking scan and an unflagged start never touch it). Returns the path removed, None
    when there was none; one `sweep_pending_decision_cleared` event. sweephold-fix-7: race-tolerant by shape -- two processes removing at once
    (two flagged daemons honouring the same `.seeded` record) see one os.remove succeed and say the event once; the other's raises FileNotFoundError and
    returns None, saying nothing. Never raises for the other process having removed it first."""
    path = os.path.join(state_dir, CONTRASTS_SWEEP_PENDING_DECISION)
    try:
        os.remove(path)
    except FileNotFoundError:
        return None
    log_event(state_dir, {"event": "sweep_pending_decision_cleared", "dry_run": False, "path": path, "by": CONTRASTS_SWEEP_HOLD_SEEDED,
                          "note": "the operator acknowledged the released state (the .seeded record): the pending decision is resolved and the once-per-advance "
                                  "sweep runs under its rule (sweephold-fix-6)"})
    return path


def _sweephold_pending_reason(state_dir: str, rec: Mapping[str, object], written_now: bool) -> str:
    """The tick's reason under `held_by_flag_pending_decision`: fix-5's sentence, then fix-6's -- the record's path and what it carries."""
    path = os.path.join(state_dir, CONTRASTS_SWEEP_PENDING_DECISION)
    hold = os.path.join(state_dir, CONTRASTS_SWEEP_HOLD); seeded = os.path.join(state_dir, CONTRASTS_SWEEP_HOLD_SEEDED)
    if rec.get("unreadable"):
        return (f"argv --no-contrasts-sweep: the hold could not be seeded here and the pending-decision record {path} exists but cannot be read "
                f"({rec.get('unreadable')}): the once-per-advance sweep stays held (sweephold-fix-6) until the operator decides (touch {hold} to hold the "
                f"sweep, or write {seeded} to acknowledge the released state)")
    unc = _sweephold_pending_unclassified(rec)
    return (f"argv --no-contrasts-sweep: the hold could not be seeded here -- the state dir's history carries {unc.get('starts')} daemon_started, "
            f"{unc.get('ticks')} tick and {unc.get('held_events')} contrasts_sweep_held event(s) ({unc.get('first_utc')} .. {unc.get('last_utc')}) nobody can "
            f"classify (sweephold-fix-5): the once-per-advance sweep is held until the operator decides (touch {hold} "
            f"to hold the sweep, or write {seeded} to acknowledge the released state); the unresolved decision is recorded in {path} "
            f"({_sweephold_pending_standing(rec, written_now)}) and holds whatever the bounded evidence scan sees this tick (sweephold-fix-6)")


def contrasts_sweep_hold_initial(args) -> Dict[str, object]:
    """sweephold / sweephold-fix-1 / sweephold-fix-2: `--no-contrasts-sweep` is the hold's
    INITIAL state only, applied ONCE per state dir and PERSISTED. The first flagged start writes the hold file (when absent) and the record
    `<state-dir>/contrasts_sweep_hold.seeded`; a later start -- a unit restart, or the daemon's own re-exec on code drift, which keeps sys.argv --
    finds the record and writes nothing, so a hold the operator released (rm) stays released. fix-2: with NO record, seeding happens only when
    the state dir carries no evidence of an earlier daemon on sweephold code (`sweephold_evidence`); with evidence -- the parent version seeded
    here and never wrote a record, say -- NOTHING is written and the decision is the operator's (`skipped: ambiguous_legacy_state`, the evidence
    named: touch the hold file to hold, write the record to acknowledge the released state). A file already present on a first flagged start
    with no evidence is left as it is (the record is still written). Without the flag nothing is written; a --dry-run daemon writes nothing at
    all (an audit run must not leave a hold or a seeding behind for the live daemon: its flag holds the sweep for that process only). Returns
    the decision, said on `daemon_started`: ``{seeded_now, seeded_before, hold_present, skipped, evidence[, operator_decision, pending_decision,
    pending_decision_cleared]}``. sweephold-fix-6: with no record, unclassified history is PERSISTED as `<state-dir>/contrasts_sweep_pending_decision`
    the first time it is derived (here or at a tick), and a standing record is honoured whatever the scan sees now -- a flagged start never seeds
    over it (reported as `pending_decision` {path, record, written_now} and named in `operator_decision`); the operator's `.seeded` record removes it
    (`pending_decision_cleared`: the path)."""
    state_dir = getattr(args, "state_dir", None)
    out: Dict[str, object] = {"seeded_now": None, "seeded_before": None, "hold_present": bool(state_dir) and os.path.exists(os.path.join(state_dir, CONTRASTS_SWEEP_HOLD)),
                              "skipped": None, "evidence": []}
    if getattr(args, "contrasts_sweep", True):
        out["skipped"] = "no_flag"; return out
    if not state_dir:
        out["skipped"] = "no_state_dir"; return out
    if getattr(args, "dry_run", False):
        out["skipped"] = "dry_run"; return out                 # a dry run writes no state: its flag holds the sweep for that process only (contrasts_sweep_for)
    before = contrasts_sweep_hold_seeding(state_dir)
    if before is not None:
        out["seeded_before"] = before                          # applied once already (or acknowledged by the operator): never recreate a released hold
        cleared = _sweephold_pending_clear(state_dir)          # fix-6: the operator's record resolves a pending decision an earlier start or tick persisted
        if cleared is not None:
            out["pending_decision_cleared"] = cleared
        return out
    history = sweephold_history(state_dir)
    evidence = list(history["evidence"])  # type: ignore[arg-type]
    out["dry_run_events_skipped"] = history["dry_run_events_skipped"]      # fix-3: dry-run audits in the log, excluded from the evidence, counted for the reader
    unclassified = history.get("unclassified")
    if unclassified:                                           # fix-4: history nobody can place in a start-up window -- reported, never counted, never seeded over
        out["unclassified"] = unclassified
    pending_path = os.path.join(state_dir, CONTRASTS_SWEEP_PENDING_DECISION)
    pending = contrasts_sweep_pending_decision(state_dir)      # fix-6: an unresolved decision an earlier start or tick persisted -- it binds whatever the scan sees now
    written_now = False
    if pending is None and unclassified:                       # fix-6: the FIRST derivation of "pending" on this state dir is persisted before anything reports it
        pending, written_now = _sweephold_pending_write(state_dir, history)   # fix-7: (the record that stands, whether THIS process published it); (None, False) when it vanished under the publish
    if pending is not None:
        out["pending_decision"] = {"path": pending_path, "record": pending, "written_now": written_now}
    decision = (f"touch {os.path.join(state_dir, CONTRASTS_SWEEP_HOLD)} to hold the sweep, or write {os.path.join(state_dir, CONTRASTS_SWEEP_HOLD_SEEDED)} "
                "(any JSON object, e.g. {\"acknowledged_by\": \"<who>\"}) to acknowledge the released state; "
                + (f"the unresolved decision is recorded in {pending_path} ({_sweephold_pending_standing(pending, written_now)}) and holds the once-per-advance "
                   "sweep in every flagged start until one of them exists (sweephold-fix-6); neither file was written" if pending is not None else "nothing was written"))
    if evidence:                                               # fix-2: a sweephold-era daemon ran here and left no record (the parent version): not ours to decide
        out["skipped"] = SWEEPHOLD_SKIP_AMBIGUOUS; out["evidence"] = evidence
        out["operator_decision"] = decision
        return out
    if unclassified or pending is not None:
        u = unclassified or _sweephold_pending_unclassified(pending)   # fix-6: the scan may no longer see them (evicted from its bounded window); the record does
        out["skipped"] = SWEEPHOLD_SKIP_UNCLASSIFIED
        out["operator_decision"] = (f"{u.get('starts')} daemon_started event(s), {u.get('ticks')} tick(s) and {u.get('held_events')} contrasts_sweep_held event(s) "   # type: ignore[union-attr]
                                    f"carrying contrasts_sweep_hold between {u.get('first_utc')} and {u.get('last_utc')} cannot be classified -- an old start-up "   # type: ignore[union-attr]
                                    "without an affirmative discriminator (a dry_run stamp, or a seeding that wrote a file) and the ticks in its window, or no "
                                    "start-up window in the scanned log (sweephold-fix-4 / fix-5)"
                                    + ("" if unclassified else " -- as recorded when the decision was first derived; the bounded scan no longer sees them")
                                    + ": they are not counted as evidence, the flag is not applied over "
                                    "them, and the once-per-advance sweep is held in this process and every later flagged one (sweephold-fix-6: the decision is "
                                    "persisted) until you decide; " + decision)
        return out
    path = os.path.join(state_dir, CONTRASTS_SWEEP_HOLD)
    os.makedirs(state_dir, exist_ok=True)
    wrote = not os.path.exists(path)
    if wrote:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(CONTRASTS_SWEEP_HOLD_ARGV_REASON + "\n")
    rec = {"utc": utc_now(), "daemon_sha": IMPORTED_CODE_SHA, "pid": os.getpid(), "wrote_file": wrote, "reason": CONTRASTS_SWEEP_HOLD_ARGV_REASON,
           "note": "the argv flag's initial state was applied here once; later starts and re-execs with the flag write nothing (a released hold stays released)"}
    with open(os.path.join(state_dir, CONTRASTS_SWEEP_HOLD_SEEDED), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, sort_keys=True); fh.write("\n")
    out["seeded_now"] = path if wrote else None
    out["seeding_recorded"] = True
    out["hold_present"] = True
    return out


#: sweephold-fix-1: the statuses `contrasts_sweep_for` returns and the tick reports under `contrasts_sweep_hold.sweep` -- what the sweep DID this tick
SWEEP_STATUS_RAN = "ran"; SWEEP_STATUS_HELD_FILE = "held_by_file"; SWEEP_STATUS_HELD_DRY_FLAG = "held_by_flag_dry_run"
SWEEP_STATUS_SKIPPED = "skipped_once_per_advance"; SWEEP_STATUS_FAILED = "failed"; SWEEP_STATUS_NO_STATE = "no_state_dir"
SWEEP_STATUS_OPERATOR_HELD = "operator_held"; SWEEP_STATUS_SCHEMA_HELD = "schema_held"; SWEEP_STATUS_DISABLED = "contrasts_disabled"
SWEEP_STATUS_HELD_FLAG_PENDING = "held_by_flag_pending_decision"   # sweephold-fix-5: flagged, nothing seeded, unclassifiable history -- held until the operator decides
_HOLD_UNREAD = object()


def _contrasts_sweep_pending(args) -> Optional[Dict[str, object]]:
    """sweephold-fix-5 / fix-6: the tick's pending-decision hold. ``{"reason", "path", "record", "written_now"}`` while the
    once-per-advance sweep is HELD on a flagged start whose hold was never seeded here; ``{"cleared": <path>, "by": <.seeded>}`` when the operator's
    `.seeded` record stands and a pending record was removed for it this tick; else None. Reached only when `--no-contrasts-sweep` is set, this is
    not a dry run (a dry run holds by its own status and derives nothing), there is a state dir, and the caller's one snapshot of the hold file
    found it absent. fix-5: with NO `.seeded` record (the operator has neither acknowledged the released state nor has a first start seeded here)
    and UNCLASSIFIED history -- an old start-up that states neither dry run nor real and its ticks, or ticks no window places --
    `contrasts_sweep_hold_initial` went to the operator-decision path and wrote nothing, and the flag's intent (a hold) stands until the operator
    decides. fix-6: that derivation came from the bounded scan EVERY tick, so stamped ticks pushing the unclassified events out of the tail window
    released the sweep with no decision made. Now the persisted record `<state-dir>/contrasts_sweep_pending_decision` is the authority: present ->
    held, whatever the scan sees (the scan is not even read); absent -> the scan is read and, on unclassified history, the record is written (the
    tick's first derivation) and the sweep held. The `.seeded` record clears it (the only release); the hold file is the caller's business (it
    supersedes this while present). A READABLE history with no unclassified events and no record (fix-2's upgrade path: a real predecessor seeded
    a file and the operator released it) holds nothing -- the release stands and the sweep runs, as fix-2 pinned. An unflagged daemon is never
    bound by the record and never removes it."""
    state_dir = getattr(args, "state_dir", None)
    if getattr(args, "contrasts_sweep", True) or not state_dir or getattr(args, "dry_run", False):
        return None
    if contrasts_sweep_hold_seeding(state_dir) is not None:
        cleared = _sweephold_pending_clear(state_dir)
        return {"cleared": cleared, "by": CONTRASTS_SWEEP_HOLD_SEEDED} if cleared is not None else None
    rec = contrasts_sweep_pending_decision(state_dir)
    written_now = False
    if rec is None:
        history = sweephold_history(state_dir)
        unc = history.get("unclassified")
        if not isinstance(unc, dict) or not unc:
            return None
        rec, written_now = _sweephold_pending_write(state_dir, history)       # fix-7: (the record that stands, whether THIS process published it)
        if rec is None:
            return None                                            # fix-7: the record vanished under the publish (a .seeded honoured that instant, or an rm): nothing pending this tick
    return {"reason": _sweephold_pending_reason(state_dir, rec, written_now), "path": os.path.join(state_dir, CONTRASTS_SWEEP_PENDING_DECISION),
            "record": rec, "written_now": written_now}


def contrasts_sweep_flag_pending(args) -> Optional[str]:
    """sweephold-fix-5: the reason the once-per-advance sweep is HELD on a flagged start whose hold was never seeded here, else None --
    `_contrasts_sweep_pending`'s reason (fix-6: derived from the persisted record, which the first derivation writes)."""
    d = _contrasts_sweep_pending(args)
    return str(d["reason"]) if d is not None and d.get("reason") is not None else None


def contrasts_sweep_for(args, summary: Dict[str, object], hold_reason=_HOLD_UNREAD) -> str:
    """The BOUNDED re-emission sweep (`emit_contrasts.sweep_noncurrent`) over EVERY native run of the daemon's campaign --
    every model and view, not this daemon's one model -- whose filed contrast's anchor is no longer current under the paper's rules, run
    ONCE per advance of the daemon's code: when IMPORTED_CODE_SHA differs from the sha the last sweep recorded in <state>/contrasts_sweep.json
    (the same trigger that releases a quarantine), never every tick (migration 0021 measured ~3,900 frame rows losing their contrast columns
    to retired anchors across models; the per-tick pass above stays scoped to this daemon's model, and the sweep leaves this model's FIRST
    emissions to it). The tick summary carries the counts (`contrasts_sweep`: due / emitted / no_anchor / cap_mismatch / deferred / first /
    retired, dry_run, the shas) and one `sweep_noncurrent` event carries the per-cell table; a failing sweep is one `sweep_noncurrent_failed`
    event and never stops the loop. A dry-run daemon sweeps dry (counts only, nothing written) and records no mark, so the next real tick
    sweeps; without a state dir there is no mark and no sweep. sweephold: while `<state-dir>/contrasts_sweep_hold` exists the sweep is
    HELD -- this returns before any registry read, writes no mark, and the summary block is ``{"held_by_file": <reason>}``; the argv flag is not
    consulted (it only seeds the file once, on start: `contrasts_sweep_hold_initial`). ``hold_reason`` is the
    tick's ONE snapshot of the file (read before the contrasts block; a flip during the per-tick pass is seen next tick, never mid-decision) --
    this reads the file itself only when called without one. Returns what the sweep did (the SWEEP_STATUS_* vocabulary), which the tick reports
    beside the snapshot. The pending-decision hold (`_contrasts_sweep_pending`) is read from its persisted record and the
    tick carries the record under `contrasts_sweep_pending_decision` ({path, record, written_now}; or {path, cleared, by} on the tick the operator's
    `.seeded` record resolved it)."""
    if not args.state_dir:
        return SWEEP_STATUS_NO_STATE
    held_reason = contrasts_sweep_hold(args.state_dir) if hold_reason is _HOLD_UNREAD else hold_reason
    if held_reason is not None:
        summary["contrasts_sweep"] = {"held_by_file": held_reason}
        return SWEEP_STATUS_HELD_FILE
    if args.dry_run and not getattr(args, "contrasts_sweep", True):
        reason = CONTRASTS_SWEEP_HOLD_ARGV_REASON + " (dry run: no file written)"           # the one place the flag is read per tick: a dry run seeds no file
        summary["contrasts_sweep"] = {"held_by_file": reason}
        return SWEEP_STATUS_HELD_DRY_FLAG
    pending = _contrasts_sweep_pending(args)                                                 # sweephold-fix-5 / fix-6: flagged, nothing seeded, the pending decision persisted
    if pending is not None and pending.get("reason") is not None:
        summary["contrasts_sweep"] = {"held_by_flag_pending_decision": pending["reason"]}
        summary["contrasts_sweep_pending_decision"] = {"path": pending["path"], "record": pending["record"], "written_now": pending["written_now"]}
        return SWEEP_STATUS_HELD_FLAG_PENDING
    if pending is not None:                                                                  # fix-6: the operator's .seeded record resolved it this tick -- said once, the sweep proceeds
        summary["contrasts_sweep_pending_decision"] = {"path": pending["cleared"], "cleared": True, "by": pending["by"]}
    mark_path = os.path.join(args.state_dir, CONTRASTS_SWEEP_MARK)
    try:
        with open(mark_path, encoding="utf-8") as fh:
            mark = json.load(fh)
    except (OSError, ValueError):
        mark = {}
    if str(mark.get("daemon_sha") or "") == IMPORTED_CODE_SHA and not args.dry_run:
        return SWEEP_STATUS_SKIPPED
    try:
        # the daemon's own model's FIRST emissions are the per-tick pass's (just above); the sweep takes the retired anchors of every model and
        # the first emissions of the other models
        counts = EC.sweep_noncurrent(registry_root=args.registry_root, campaign=args.campaign, spool=args.spool, dry_run=args.dry_run, quiet=True,
                                     first_skip_models=list(args.model_id or []))
    except Exception as exc:  # noqa: BLE001 -- a sweep never stops the grading loop
        log_event(args.state_dir, {"event": "sweep_noncurrent_failed", "error": f"{type(exc).__name__}: {exc}"})
        return SWEEP_STATUS_FAILED
    block: Dict[str, object] = {k: counts["totals"][k] for k in EC.SWEEP_COUNT_KEYS}   # type: ignore[index]
    block.update(dry_run=bool(args.dry_run), daemon_sha=IMPORTED_CODE_SHA, was_daemon_sha=mark.get("daemon_sha"))
    summary["contrasts_sweep"] = block
    # the event is `sweep_noncurrent`, not `contrasts_*`: a `contrasts_*` event other than `contrasts` is a defect of the per-tick pass (its test pins that)
    log_event(args.state_dir, {"event": "sweep_noncurrent", **block, "readonly": counts["readonly"], "cells": counts["cells"],
                               "skipped": counts["totals"].get("skipped"), "other": counts["totals"].get("other"),   # type: ignore[union-attr]
                               "note": "the bounded re-emission sweep over every native run of the campaign whose filed contrast anchor is no longer "
                                       "current under the paper's rules; run once per daemon code advance"})
    if not args.dry_run:
        with open(mark_path, "w", encoding="utf-8") as fh:
            json.dump({"daemon_sha": IMPORTED_CODE_SHA, "utc": utc_now(), "counts": block, "sweephold_era": True}, fh, indent=2, sort_keys=True); fh.write("\n")   # fix-2: evidence for the upgrade path
    return SWEEP_STATUS_RAN


def supersession_reconcile_pass(args, summary: Dict[str, object]) -> None:
    """A complete judgment standing beside the same judge's PARTIALS -- the state a grader dying
    between its completion and its supersessions left before the atomic batch, or any older re-judge -- is repaired every tick:
    `worklist.reconcile_supersessions` writes the same superseded records through the same atomic writer (one record file per (run, judge);
    a partial whose record is already on the spool or whose status folded is skipped, so the pass is idempotent), one `supersession_reconciled`
    event per pair naming the ids. Nothing under --dry-run (reported only); a failure is a `problem`, never the tick's. A
    pair the worklist REFUSED by name (its standing partial's judgment_id is reproducible from none of the registry's identity forms -- plain,
    claimed_by_session + claimed_utc, producer_session + claimed_utc -- so a same-id superseded record for it would be refused) is one
    `supersession_refused` event naming the exception, counted under `supersessions_refused`, never under `supersessions_reconciled`; nothing was
    written for it and it is said again every tick."""
    if not args.registry_root:
        return
    try:
        done = RW.reconcile_supersessions(args.registry_root, model_ids=args.model_id or None, campaign=args.campaign, session=getattr(args, "grader_session", None) or None,
                                          script="scripts/registry/grade_daemon.py", dry_run=bool(args.dry_run))
    except Exception as exc:  # noqa: BLE001 -- the pass must never take the tick down; it is said and tried again next tick
        probs = summary.setdefault("problems", [])
        if isinstance(probs, list):
            probs.append(f"supersession_reconcile: {type(exc).__name__}: {exc}")
        log_event(args.state_dir, {"event": "problem", "kind": "supersession_reconcile", "error": f"{type(exc).__name__}: {exc}"})
        return
    for d in done:
        if d.get("refused"):
            log_event(args.state_dir, {"event": "supersession_refused", "run_id": d["run_id"], "judge_id": d["judge_id"], "complete_id": d["complete_id"],
                                       "partial_ids": list(d.get("refused_partial_ids") or []), "refused": d["refused"], "dry_run": d["dry_run"],
                                       "note": "g41 (D1949 / D1954): the standing partial's judgment_id is reproducible from none of the registry's identity forms "
                                               "(plain / claimed_by_session + claimed_utc / producer_session + claimed_utc), so a same-id superseded record for it "
                                               "would be refused (D639): nothing written; the complete stands beside a still-current partial until that partial is "
                                               "retired by another path"})
            continue
        log_event(args.state_dir, {"event": "supersession_reconciled", "run_id": d["run_id"], "judge_id": d["judge_id"], "complete_id": d["complete_id"],
                                   "partial_ids": d["partial_ids"], "path": d["path"], "dry_run": d["dry_run"],
                                   "note": "a complete judgment stood beside the same judge's partial(s) with no supersession on record (a grader died between "
                                           "its completion and its supersessions, or a pre-g39 re-judge): the superseded records are written as one atomic file"})
    summary["supersessions_reconciled"] = int(summary.get("supersessions_reconciled", 0)) + sum(1 for d in done if not d["dry_run"] and not d.get("refused"))
    summary["supersessions_refused"] = int(summary.get("supersessions_refused", 0)) + sum(1 for d in done if d.get("refused"))


def tick(args, templates: Mapping[str, str], extra: Mapping[str, str], mirror_ssh: Mapping[str, str], store_root_map: Optional[P.StoreRootMap],
         extra_families: Optional[Mapping[str, str]] = None) -> Dict[str, int]:
    summary: Dict[str, object] = {"pending": 0, "dispatched": 0, "confirm_only": 0, "unknown": 0, "refused": 0, "failed": 0, "retryable": 0, "skipped_low_disk": 0, "skipped_locked": 0, "skipped_identity": 0,
                                  "credential": 0,    # Runs whose grader's credential the API rejected this tick (the family held, never quarantined, never counted)
                                  "rows_failed": 0, "rows_pending": 0, "rows_retried": 0, "rows_cleared": 0, "rows_exhausted": 0,   # the derived-rows ledger
                                  "schema_held": 0,   # runs whose grader's claim refused the registry's schema this tick (held transient, never quarantined)
                                  "not_folded": 0,    # Runs whose grader's claim was refused as not folded this tick (held transient, never quarantined)
                                  "claim_retry_later": {},   # The same runs by the refusal CLASS that held them ({ClaimNotFolded: n, ClaimPendingCorrection: m, ...})
                                  "identity_changed": 0,   # runs whose grader refused on the registry moving under it this tick (held transient, re-selected)
                                  "identity_held": 0,      # runs of a group HELD after IDENTITY_HOLD_AFTER unchanged identity refusals (skipped this tick)
                                  "supersessions_reconciled": 0,   # (run, judge) pairs whose standing partials were superseded by this tick's reconciliation
                                  "supersessions_refused": 0,      # pairs the reconciliation refused by name (a partial no identity form reproduces); nothing written
                                  "dry_run": bool(args.dry_run),   # sweephold-fix-3: a dry run's tick is stamped -- never evidence of a real sweephold-era daemon
                                  "daemon_sha": IMPORTED_CODE_SHA}   # the code this tick ran (the checkout's HEAD is check_code_drift's business)
    # The SCHEMA PREFLIGHT, before ANY work -- no rows retry, no claim reconciliation, no pending read, no spooled
    # or quarantine pass, no mirror, no dispatch, no contrast, no prune: while the live registry's migration version or shape is not the code's
    # the tick is this check and its summary line (`held: true, hold: schema`), and it releases by itself when the live schema is the code's
    gap = SCHEMA_GATE.check(args.registry_root, args.state_dir)
    if gap is not None:
        summary["held"] = True; summary["hold"] = "schema"; summary["held_reason"] = schema_hold_reason(gap)
        summary["schema"] = {k: gap.get(k) for k in ("kind", "have", "want", "n_drift")}; summary["schema_hold_since_utc"] = gap.get("since_utc")
        summary["problems"] = [f"schema_hold: {summary['held_reason']}"]
        hold_now = contrasts_sweep_hold(args.state_dir)                                       # sweephold-fix-1: the hold status on a schema-held tick too
        summary["contrasts_sweep_hold"] = {"held": hold_now is not None, "reason": hold_now, "sweep": SWEEP_STATUS_SCHEMA_HELD}
        log_event(args.state_dir, {"event": "tick", **summary})
        return summary
    mirror_roots = mirror_roots_of(args.store_root_map)
    # The derived-rows retries FIRST -- a run whose judgments are filed but whose rows step failed is not pending any
    # more, so nothing below would reach it; and the claim reconciliation right after settles a dead daemon's dispatch (retiring its
    # still-pending copies), so an open item's retry runs on the retained copy before anything can retire it (mirror_settle itself keeps a
    # copy an open item references, whichever instance settles)
    fam_holds = family_holds(args.state_dir)                 # the per-family holds, read at the top -- before any work for a family
    rows_retry_pass(args, summary, held=dispatch_hold(args.state_dir) is not None, held_families=fam_holds)
    rc_claims = mirror_reconcile_claims(args.state_dir, args.registry_root, args.campaign) if (args.state_dir and not args.dry_run) else {"reconciled": 0, "deferred": 0}
    summary["mirror_claims_reconciled"] = rc_claims["reconciled"]; summary["mirror_claims_deferred"] = rc_claims["deferred"]
    identity_ledger_load(args.state_dir)                                                           # the identity-hold ledger survives a re-exec
    identity_release_asked = _identity_release_requests(args.state_dir, consume=not args.dry_run)   # the operator's release file, read once per tick
    supersession_reconcile_pass(args, summary)                                                     # a complete beside the same judge's partials
    rows = pending_rows(args.registry_root, args.model_id, args.campaign, store_root_map, extra_families, grader_session=getattr(args, "grader_session", None))
    refusal_clear(args.state_dir, [rid for rid in list(_REFUSED_RUNS) if rid not in {str(r["run_id"]) for r in rows}], "left the pending set")
    not_folded_clear(args.state_dir, [rid for rid in list(_NOT_FOLDED) if rid not in {str(r["run_id"]) for r in rows}], "left the pending set")
    identity_changed_clear(args.state_dir, [rid for rid in list(_IDENTITY_CHANGED) if rid not in {str(r["run_id"]) for r in rows}], "left the pending set")
    if args.families:
        rows = [r for r in rows if r["family"] in args.families]
    summary["pending"] = len(rows)
    # A held family's rows leave the tick HERE -- before the spooled / quarantine passes see them as pending rows and before any
    # grouping, mirror or dispatch -- so they stay pending / quarantined exactly as they are (the quarantine ledger's releases still run below;
    # a released run of a held family is not in `rows`, so it is skipped this tick, never re-quarantined and never counted as failed)
    fam_counts: Dict[str, int] = {}
    for r in rows:
        if str(r["family"]) in fam_holds:
            fam_counts[str(r["family"])] = fam_counts.get(str(r["family"]), 0) + 1
    rows = [r for r in rows if str(r["family"]) not in fam_holds]
    summary["held_families"] = sorted(fam_holds); summary["held_family_runs"] = sum(fam_counts.values())
    family_holds_report(args.state_dir, fam_holds, fam_counts)
    # A reclaim row whose claim's completion is already on the spool gets its hold back BEFORE the hold filter decides (a dry run writes no
    # ledger: those rows are held from this tick's report by hand)
    rebuilt = spooled_rebuild(args.state_dir, args.registry_root, rows, dry_run=args.dry_run)
    summary["spooled_rebuilt"] = len(rebuilt)
    if args.dry_run and rebuilt:
        rows = [r for r in rows if str(r["run_id"]) not in set(rebuilt)]
    # the spooled hold FIRST: a hold whose judgment folded short releases INTO the quarantine, which the quarantine pass below must then
    # see in the same tick — the two ledgers never hold one run at once, so the order changes nothing else
    rows, spooled = spooled_filter(args.state_dir, args.registry_root, rows, dry_run=args.dry_run)
    summary["spooled_hold"] = len(spooled)
    rows, held = quarantine_filter(args.state_dir, args.registry_root, rows, dry_run=args.dry_run)
    summary["quarantined"] = len(held)
    summary["problems"] = [f"dispatch_quarantine: {len(held)} pending run(s) held after a failed dispatch"] if held else []
    summary["fold_mode"] = "harvester"                       # the graders' judgments reach the DB on the harvester's cycle; a spooled-not-folded run is held, never re-dispatched
    # the own-session reclaim is reported over the rows that remain DISPATCHABLE: a run held by the quarantine or the spooled hold keeps its
    # claim and is not re-dispatched, so it is not reclaimed (a spooled run's claim must not read as a re-dispatch)
    hold = dispatch_hold(args.state_dir)                      # the operator's hold, read every tick
    summary["held"] = hold is not None
    if hold is not None:
        summary["hold"] = "operator"; summary["held_reason"] = hold          # the hold's KIND beside the flag (the schema hold is the other kind)
        log_event(args.state_dir, {"event": "held", "mid_tick": False, "hold": "operator", "reason": hold, "pending": len(rows), "run_ids": sorted(str(r["run_id"]) for r in rows),
                                   "file": os.path.join(str(args.state_dir), DISPATCH_HOLD),
                                   "note": "operator dispatch hold: every pending row left untouched, no grader dispatched, no contrast emitted; "
                                           "the spooled / quarantine bookkeeping above still ran; cleared by removing the file"})
        rows = []
    reclaim = [str(r["run_id"]) for r in rows if r.get("reclaim")]
    summary["reclaimed"] = len(reclaim)
    if reclaim:
        log_event(args.state_dir, {"event": "reclaim_own_session", "n": len(reclaim), "run_ids": sorted(reclaim), "session": getattr(args, "grader_session", None) or RW.default_session(),
                                   "note": "claimed by this daemon's graders' session and no grader of this daemon is running: the grader that claimed them "
                                           "exited before filing; re-dispatched under the same claim"})
    groups = sorted(group_rows(rows).items())
    refusals: List[Tuple[str, List[str], DispatchRefused]] = []                      # This tick's refusals, for the problem at the end
    for gi, ((family, model_key, campaign, store_root, private_run, one_eval), group) in enumerate(groups):
        run_ids = [str(r["run_id"]) for r in group]
        hold_now = dispatch_hold(args.state_dir)                 # the hold is re-read before EVERY group
        if hold_now is not None:
            _hold_stop(args.state_dir, summary, hold_now, groups[gi:], where="before this group's mirror and claim")
            break
        fam_now = family_holds(args.state_dir)                   # a family hold written DURING the tick stops that family's later groups
        if family in fam_now:
            summary["held_family_runs"] = int(summary.get("held_family_runs", 0)) + len(group)
            if family not in summary.get("held_families", []):
                summary["held_families"] = sorted(set(summary.get("held_families", [])) | {family})   # type: ignore[arg-type]
            family_holds_report(args.state_dir, fam_now, {family: len(group)}, where="before this group's mirror and claim")   # the COMPLETE map
            continue
        held_ids = [rid for rid in run_ids if _IDENTITY_CHANGED.get(rid, {}).get("held")]
        if held_ids:
            # The group is HELD after IDENTITY_HOLD_AFTER unchanged identity refusals -- no mirror, no claim, no grader; the
            # copy stays and the standing `problem` says so every tick -- until the registry moved under it or the operator released it
            why = identity_hold_release(args.state_dir, args.registry_root, run_ids, held_ids, identity_release_asked)
            if why is None:
                summary["identity_held"] = int(summary.get("identity_held", 0)) + len(group)
                continue
            log_event(args.state_dir, {"event": "dispatch_identity_hold_released", "family": family, "model_key": model_key, "campaign": campaign,
                                       "run_ids": run_ids, "held_run_ids": held_ids, "why": why,
                                       "note": "the group held on a persisting identity refusal dispatches again: " + ("the operator's release file named it"
                                               if why == "operator" else "the registered identity it refused on has changed (re-selected against the new state)")})
        if family in CONFIRM_ONLY:
            summary["confirm_only"] += len(group)
            log_event(args.state_dir, {"event": "confirm_only", "family": family, "model_key": model_key, "n": len(group),
                                       "note": "rule-scored in the job / scored on Modal: the daemon confirms is_complete, never dispatches a judge"})
            continue
        if family not in templates:
            summary["unknown"] += len(group)
            log_event(args.state_dir, {"event": "unknown_family", "family": family, "eval_ids": sorted({str(r["eval_id"]) for r in group}), "n": len(group)})
            continue
        # The dispatch's paths come FIRST -- its log, and for a family whose judged root carries
        # served-identity files the immutable per-dispatch snapshot directory beside that log, which the grader is pointed at
        # (`--served-identity-dir`) so the shared mirror root is never a grader's read path
        log_path = _dispatch_log_path(args.state_dir, family, model_key)
        identity_dir = served_identity_dir_for(log_path) if family in ROOT_IDENTITY_FILES else None
        try:
            cmd = build_command(templates[family], group, python=args.python, repo=args.repo, registry_root=args.registry_root,
                                store_root_map_specs=args.store_root_map or [], extra=extra.get(family, ""), model_key=model_key,
                                configs_root=args.configs_root, grades_root=args.grades_root, served_identity_dir=identity_dir)
        except DispatchRefused as exc:
            summary["refused"] += len(group)
            refusals.append((family, run_ids, exc))
            # said ONCE per run per (reason, detail) -- the runs of this group not yet refused for exactly this, or refused for something
            # else; a group whose every run was already said for this reason says nothing again (the tick line's `refused` and the `problem` below
            # carry the standing count), so a watcher sees a refusal the first tick it happens and the log does not repeat it every tick
            fresh = [rid for rid in run_ids if _REFUSED_RUNS.get(rid, (None, None, None))[:2] != (exc.reason, exc.detail)]
            if fresh:
                now = utc_now()
                for rid in fresh:
                    _REFUSED_RUNS[rid] = (exc.reason, exc.detail, exc.path)
                log_event(args.state_dir, {"event": "dispatch_refused", "family": family, "model_key": model_key, "campaign": campaign, "n": len(group),
                                           "run_ids": run_ids, "new_run_ids": fresh, "store_root": store_root or None, "private_run": private_run or None,
                                           "eval_ids": sorted({str(r["eval_id"]) for r in group}), "reason": exc.reason, "detail": exc.detail,
                                           "config_path": exc.path, "daemon_sha": IMPORTED_CODE_SHA, "first_seen_utc": now,
                                           "note": "left pending; nothing claimed, no driver run; said once per run per reason (dispatch_refusal_cleared when "
                                                   "the run dispatches or leaves the pending set); the tick line's `refused` and a `problem` carry the standing count"})
            continue
        # the group's mirror directories are held exclusively from before the copy through the dispatch and the settlement (an overlapping
        # instance — a `--once`, a replacement daemon — skips the group this tick instead of copying over or retiring what this one reads)
        with mirror_locks([] if args.dry_run else [local for _r, _a, _re, local in mirror_targets(group, mirror_ssh)]) as held:
            if held.busy:
                summary["skipped_locked"] += len(group)
                log_event(args.state_dir, {"event": "mirror_skipped_locked", "family": family, "model_key": model_key, "run_ids": run_ids, "dirs": held.busy,
                                           "note": "another instance (or an orphaned grader) holds these mirror directories: the group waits for a later tick"})
                continue
            mirrored = mirror_stores(group, mirror_ssh, args.rsync_bin, args.dry_run, args.state_dir, mirror_roots, getattr(args, "mirror_min_free_bytes", MIRROR_MIN_FREE_BYTES_DEFAULT),
                                     identity_snapshot_dir=identity_dir)
            if mirrored is None:                       # the GUARD: below the free-space floor nothing is mirrored and nothing dispatched this tick
                summary["skipped_low_disk"] += len(group)
                continue
            blocked_roots = identity_mirror_blocked(mirrored)
            if blocked_roots:                          # the judged root's identity files did not transfer -- no dispatch on a partial identity set
                summary["skipped_identity"] = int(summary.get("skipped_identity", 0)) + len(group)
                log_event(args.state_dir, {"event": "mirror_identity_failed", "family": family, "model_key": model_key, "run_ids": run_ids, "roots": blocked_roots,
                                           "rcs": {str(m.get("identity_root")): m.get("identity_rc") for m in mirrored if m.get("identity_rc") not in (None, 0)},
                                           "note": "the identity files of the judged root did not transfer this tick: the group is not dispatched (a grader must never "
                                                   "read a partial or stale identity set); the stores stay mirrored, retried next tick"})
                continue
            if identity_dir is not None and store_root and not args.dry_run and not any(m.get("identity_root") for m in mirrored):
                # A judged root read where it is (no mirror alias for its substrate; a path the map left unchanged) had no identity
                # transfer to snapshot under -- taken here, under the same root identity lock, so the grader's read path is its own directory
                # either way (a root that is nobody's mirror target still gets no shared read)
                with root_identity_lock(str(store_root)) as taken:
                    snapped = snapshot_served_identity(str(store_root), identity_dir, ROOT_IDENTITY_FILES[family], args.state_dir, family) if taken else None
                if snapped is None:
                    summary["skipped_identity"] = int(summary.get("skipped_identity", 0)) + len(group)
                    log_event(args.state_dir, {"event": "mirror_identity_failed", "family": family, "model_key": model_key, "run_ids": run_ids, "roots": [str(store_root)],
                                               "rcs": {str(store_root): "snapshot failed" if taken else "locked: another instance is transferring this root's identity files right now"},
                                               "note": "the dispatch's served-identity snapshot could not be taken: the group is not dispatched this tick"})
                    continue
            log_event(args.state_dir, {"event": "dispatch", "family": family, "model_key": model_key, "campaign": campaign, "n": len(group),
                                       "run_ids": run_ids, "store_root": store_root or None, "private_run": private_run or None,
                                       "served_identity_dir": identity_dir,
                                       "cmd": " ".join(shlex.quote(c) for c in cmd), "dry_run": args.dry_run})
            refusal_clear(args.state_dir, run_ids, "dispatched", family)              # A run refused before now dispatches
            if args.dry_run:
                summary["dispatched"] += len(group)
                continue
            claim_id = mirror_claim_dispatch(args.state_dir, family=family, model_key=model_key, campaign=args.campaign, run_ids=run_ids, mirrored=mirrored)
            grader_env = dict(os.environ)
            grader_env.setdefault(GRADER_LOCK_TIMEOUT_ENV, str(getattr(args, "grader_lock_timeout_s", GRADER_LOCK_TIMEOUT_S_DEFAULT)))
            dispatch_utc = utc_now()                   # the hold binds to the judgment records this grader writes from here on
            hold_now = dispatch_hold(args.state_dir)   # re-read once more, right before the grader starts (a long group A may have run for hours)
            if hold_now is not None:
                mirror_settle(args.state_dir, args.registry_root, family=family, model_key=model_key, campaign=args.campaign, run_ids=run_ids, mirrored=mirrored,
                              rc=None, reason="operator dispatch hold appeared after the mirror and claim, before the grader started")
                mirror_release_dispatch(args.state_dir, claim_id)
                _hold_stop(args.state_dir, summary, hold_now, groups[gi:], where="after this group's mirror and claim, before its grader")
                break
            fam_now = family_holds(args.state_dir)     # the FAMILY holds re-read at this gate too -- a `dispatch.hold.<family>`
            if family in fam_now:                      # written while mirror_stores copied this group's stores was seen by neither earlier read
                mirror_settle(args.state_dir, args.registry_root, family=family, model_key=model_key, campaign=args.campaign, run_ids=run_ids, mirrored=mirrored,
                              rc=None, reason="operator family hold appeared after the mirror and claim, before the grader started")
                mirror_release_dispatch(args.state_dir, claim_id)
                summary["held_family_runs"] = int(summary.get("held_family_runs", 0)) + len(group)
                summary["held_families"] = sorted(set(summary.get("held_families", [])) | {family})   # type: ignore[arg-type]
                family_holds_report(args.state_dir, fam_now, {family: len(group)}, where="after this group's mirror and claim, before its grader")
                continue                               # this family only: the other families' groups go on (no rows item opened, no grader run)
            dispatch_lock = dispatch_lock_for(log_path)          # taken here, inherited by the grader, free only once both are gone
            dfd = dispatch_lock_take(dispatch_lock)
            rows_items_open(args.state_dir, group=group, family=family, model_key=model_key, campaign=campaign, store_root=store_root or None, snapshot_dir=identity_dir,
                            cmd=cmd, dirs=[str(m.get("local") or "") for m in mirrored if m.get("local")], dispatch_utc=dispatch_utc, log_path=log_path,
                            dispatch_lock=dispatch_lock if dfd is not None else None)
            try:
                # the grader INHERITS the store locks and the dispatch lock (pass_fds): they live as long as it runs, whatever happens to this
                # daemon; its output goes to the dispatch log so a failure's tail can be recorded with the run; its registry claims WAIT for the
                # harvester's lock (GRADER_LOCK_TIMEOUT_ENV) instead of dying at the first held poll
                with open(log_path, "ab") as log_fh:
                    proc = subprocess.run(cmd, cwd=args.repo, timeout=args.driver_timeout_s, pass_fds=tuple(held.fds) + ((dfd,) if dfd is not None else ()),
                                          stdout=log_fh, stderr=subprocess.STDOUT, env=grader_env)
                rc = proc.returncode
            except (OSError, subprocess.SubprocessError) as exc:
                rc = f"error: {exc}"
            finally:
                if dfd is not None:
                    os.close(dfd)                                # this daemon's hold ends with the grader; a grader that outlives us keeps its own
            tail = _tail(log_path) if rc != 0 else ""
            # This group's tags -- its private run and run ids -- plus the TARGET the family's grader
            # was dispatched with (`target_for`: the `sdfbase_full22_<TAG>` suffix for the sampled / multi-turn / BoN lanes, the padded key, else the model
            # key) -- grade_full22_native tags its wrapper `[{args.target}]`, which is neither the private run nor a run id, so every ClaimNotPending /
            # ClaimNotFolded / identity-change refusal from a native-grader group read as FOREIGN -> failed -> the quarantine, and the admission was
            # defeated for exactly the graders it was widened for. The tags are the DISPATCH's to compose; the admission stays family-blind.
            group_tags = frozenset(t for t in [str(private_run or ""), target_for(family, private_run, model_key)] + run_ids if t)
            kind = dispatch_failure_kind(rc, tail, group_tags=group_tags)
            if kind != "not_folded":
                not_folded_clear(args.state_dir, run_ids, f"the dispatch's outcome is {kind}", family)   # Another outcome ends the standing refusal
            if kind != "identity_changed":
                identity_changed_clear(args.state_dir, run_ids, f"the dispatch's outcome is {kind}", family)   # likewise
            # the cap ruling (2026-10-13 23:29): a retryable failure is charged per run by what the dispatch did -- read from the spool
            # the judgment records written since `dispatch_utc` (filed / attempted-and-incomplete / never reached)
            progress = dispatch_progress(args.registry_root, campaign, group, dispatch_utc) if kind == "retryable" else None
            n_retry = retryable_count(args.state_dir, run_ids, failed=(kind == "retryable"), progress=progress)
            if kind == "retryable" and n_retry >= RETRYABLE_CAP:
                kind = "failed"; tail = tail + f"\n[grade-daemon] {n_retry} consecutive retryable failures (cap {RETRYABLE_CAP}): quarantined — the upstream shed is not clearing"
            # The rows items settle from the SPOOL (which runs this dispatch filed a complete judgment for) before the copies are
            # settled -- a run whose rows are still owed keeps its item, and the stores it graded are kept for the retry
            rows_st = rows_items_settle(args.state_dir, args.registry_root, group=group, family=family, campaign=campaign, rc=rc, tail=tail, dispatch_utc=dispatch_utc, log_path=log_path)
            log_event(args.state_dir, {"event": "dispatched", "family": family, "model_key": model_key, "rc": rc, "n": len(group), "log": log_path, "outcome": kind,
                                       **({"rows_pending": rows_st["pending"]} if rows_st["pending"] else {}),
                                       **({"stderr_tail": tail, "terminal_exception": terminal_exception_line(tail)} if rc != 0 else {})})
            st = mirror_settle(args.state_dir, args.registry_root, family=family, model_key=model_key, campaign=args.campaign, run_ids=run_ids, mirrored=mirrored,
                               rc=rc, retryable=(kind in ("retryable", "schema_drift", "not_folded", "identity_changed", "credential") or bool(rows_st["pending"])))   # a schema, not-folded, identity or credential refusal keeps the copy too
            mirror_release_dispatch(args.state_dir, claim_id)
            summary["mirror_retired_dirs"] = int(summary.get("mirror_retired_dirs", 0)) + st["retired"]  # type: ignore[arg-type]
            summary["mirror_retired_bytes"] = int(summary.get("mirror_retired_bytes", 0)) + st["retired_bytes"]  # type: ignore[arg-type]
            if rc == 0:
                summary["dispatched"] += len(group)        # complete: the grader filed AND (for a derived-rows family) its rows items cleared (rows_items_settle)
                spooled_hold_add(args.state_dir, group=group, family=family, model_key=model_key, campaign=campaign, log_path=log_path,
                                 registry_root=args.registry_root, dispatch_utc=dispatch_utc)
            else:
                rest = group
                if rows_st["pending"]:
                    # the grader FILED these runs' complete judgments and then failed (its rows step, or anything after the filing):
                    # not a quarantine (the run leaves the pending view at the fold and a quarantine entry would be dropped) -- the derived-rows
                    # ledger owns the retry; the spooled hold keeps the run from a re-dispatch until the fold shows, exactly as after a clean exit
                    filed = [r for r in group if str(r["run_id"]) in rows_st["pending"]]
                    rest = [r for r in group if str(r["run_id"]) not in rows_st["pending"]]
                    summary["rows_failed"] = int(summary.get("rows_failed", 0)) + len(filed)
                    spooled_hold_add(args.state_dir, group=filed, family=family, model_key=model_key, campaign=campaign, log_path=log_path,
                                     registry_root=args.registry_root, dispatch_utc=dispatch_utc)
                    log_event(args.state_dir, {"event": "dispatch_rows_failed", "family": family, "model_key": model_key, "campaign": campaign, "n": len(filed),
                                               "run_ids": sorted(rows_st["pending"]), "rc": rc, "dispatch_log": log_path, "snapshot_dir": identity_dir, "attempt": 1, "cap": ROWS_ATTEMPT_CAP,
                                               "terminal_exception": terminal_exception_line(tail), "ledger": _rows_path(str(args.state_dir)),
                                               "note": "the judgments are filed; the derived rows are not: the dispatch is NOT complete -- retried from the retained "
                                                       "snapshot on the next ticks (rows_retry_pass), the graded stores kept until the rows item clears"})
                if not rest:
                    pass
                elif kind == "schema_drift":
                    # the grader's claim refused the registry's schema (a migration landed between this tick's preflight and the claim,
                    # or the code is ahead of the database): TRANSIENT and HELD -- the runs stay pending for the next tick's preflight, the copy is
                    # kept, nothing is quarantined, nothing counts toward RETRYABLE_CAP; and the gate is re-read NOW, so the groups not yet
                    # dispatched stop here instead of each failing the same way
                    term = terminal_exception_line(tail)
                    summary["schema_held"] = int(summary.get("schema_held", 0)) + len(rest)
                    log_event(args.state_dir, {"event": "dispatch_schema_held", "family": family, "model_key": model_key, "campaign": campaign, "n": len(rest),
                                               "run_ids": [str(r["run_id"]) for r in rest], "rc": rc, "dispatch_log": log_path, "terminal_exception": term,
                                               **schema_gap_of_terminal(term),
                                               "note": "the grader's registry claim refused the database: its migration version or shape is not the code's "
                                                       "(SchemaDrift). Left PENDING for the next tick's schema preflight, the copy kept; never quarantined and never "
                                                       "counted toward the retryable cap -- a completed migration releases nothing a quarantine would wait on"})
                    gap_now = SCHEMA_GATE.check(args.registry_root, args.state_dir)
                    if gap_now is not None:
                        _schema_stop(args.state_dir, summary, gap_now, groups[gi + 1:], where="after this group's grader refused the registry's schema")
                        break
                elif kind == "not_folded":
                    # the grader's claim was refused because the run's (or the judge's) row has not folded yet (pending.ClaimNotFolded; nothing
                    # written): TRANSIENT and HELD like a schema refusal -- the runs stay pending, the copy is kept, nothing is quarantined, nothing counts
                    # toward RETRYABLE_CAP; a run still not folded past NOT_FOLDED_PROBLEM_TICKS ticks is a problem every tick it stands (uncapped)
                    term = terminal_exception_line(tail); now = utc_now()
                    refusal_class = retry_later_refusal_class(term) or "ClaimNotFolded"      # Which retry-later class refused
                    summary["not_folded"] = int(summary.get("not_folded", 0)) + len(rest)
                    crl = summary.setdefault("claim_retry_later", {})
                    if isinstance(crl, dict):
                        crl[refusal_class] = int(crl.get(refusal_class, 0)) + len(rest)
                    for r in rest:
                        ent = _NOT_FOLDED.setdefault(str(r["run_id"]), {"since_utc": now, "ticks": 0})
                        ent.update({"ticks": int(ent.get("ticks") or 0) + 1, "family": family, "model_key": model_key, "message": term, "last_utc": now, "dispatch_log": log_path,
                                    "refusal_class": refusal_class})
                    ticks = max(int(_NOT_FOLDED[str(r["run_id"])]["ticks"]) for r in rest)
                    log_event(args.state_dir, {"event": "dispatch_not_folded", "family": family, "model_key": model_key, "campaign": campaign, "n": len(rest),
                                               "run_ids": [str(r["run_id"]) for r in rest], "rc": rc, "dispatch_log": log_path, "message": term, "ticks": ticks,
                                               "refusal_class": refusal_class,
                                               "since_utc": min(str(_NOT_FOLDED[str(r["run_id"])]["since_utc"]) for r in rest),
                                               "note": "the grader's registry claim was refused: the run's (or the judge's) row has not folded yet (ClaimNotFolded; nothing "
                                                       "written). Left PENDING for the next tick, the copy kept; never quarantined, never counted toward the retryable cap; "
                                                       f"a run still not folded after {NOT_FOLDED_PROBLEM_TICKS} ticks is a `problem` every tick it stands"})
                elif kind == "identity_changed":
                    # the Petri grader refused because the registry moved under it between the binding and the filing (an anchor's or an
                    # in-view run's identity, currency or registered completion changed; a selected run left the judged view): a correct, transient
                    # answer -- the runs stay pending and are re-selected next tick against the registry's new state, the copy is kept, nothing is
                    # quarantined, nothing counts toward RETRYABLE_CAP; a group still refusing past IDENTITY_CHANGED_PROBLEM_TICKS ticks is a problem
                    term = terminal_exception_line(tail); now = utc_now()
                    summary["identity_changed"] = int(summary.get("identity_changed", 0)) + len(rest)
                    fp = identity_fingerprint(args.registry_root, run_ids, term)       # the registered identity the refusal depends on
                    for r in rest:
                        ent = _IDENTITY_CHANGED.setdefault(str(r["run_id"]), {"since_utc": now, "ticks": 0})
                        ent.update({"ticks": int(ent.get("ticks") or 0) + 1, "family": family, "model_key": model_key, "message": term, "last_utc": now, "dispatch_log": log_path})
                        if ent.get("fingerprint") == fp:
                            ent["n_unchanged"] = int(ent.get("n_unchanged") or 0) + 1
                        else:
                            ent.update({"fingerprint": fp, "n_unchanged": 1})
                    ticks = max(int(_IDENTITY_CHANGED[str(r["run_id"])]["ticks"]) for r in rest)
                    n_same = max(int(_IDENTITY_CHANGED[str(r["run_id"])].get("n_unchanged") or 0) for r in rest)
                    log_event(args.state_dir, {"event": "dispatch_identity_changed", "family": family, "model_key": model_key, "campaign": campaign, "n": len(rest),
                                               "run_ids": [str(r["run_id"]) for r in rest], "rc": rc, "dispatch_log": log_path, "message": term, "ticks": ticks,
                                               "since_utc": min(str(_IDENTITY_CHANGED[str(r["run_id"])]["since_utc"]) for r in rest),
                                               "fingerprint": fp, "n_unchanged": n_same, "hold_after": IDENTITY_HOLD_AFTER,
                                               "note": "the grader refused because the registry moved under it between the binding and the filing (an identity-change "
                                                       "refusal: correct and transient, not a defect). Left PENDING and re-selected next tick against the registry's new "
                                                       "state, the copy kept; never quarantined, never counted toward the retryable cap; a group still refusing after "
                                                       f"{IDENTITY_CHANGED_PROBLEM_TICKS} ticks is a `problem` every tick it stands; after {IDENTITY_HOLD_AFTER} refusals "
                                                       "with an unchanged fingerprint the group is HELD until the fingerprint changes or the release file names it"})
                    if n_same >= IDENTITY_HOLD_AFTER:
                        # the same refusal on the same registered identity IDENTITY_HOLD_AFTER times: hold the GROUP (not the family)
                        for r in rest:
                            _IDENTITY_CHANGED[str(r["run_id"])].update({"held": True, "held_utc": now})
                        summary["identity_held"] = int(summary.get("identity_held", 0)) + len(rest)
                        log_event(args.state_dir, {"event": "dispatch_identity_held", "family": family, "model_key": model_key, "campaign": campaign, "n": len(rest),
                                                   "run_ids": [str(r["run_id"]) for r in rest], "n_refusals": n_same, "fingerprint": fp, "message": term,
                                                   "release_file": os.path.join(str(args.state_dir or ""), DISPATCH_IDENTITY_HOLD_RELEASE),
                                                   "note": f"{n_same} identity refusals with an unchanged registered identity: this group is HELD -- no mirror, no claim, "
                                                           "no grader start -- until the fingerprint changes (the registry moves: released by itself) or the release file "
                                                           "names its run ids (or `*`); the copy is kept and the problem stands every tick; other groups dispatch"})
                    identity_ledger_save(args.state_dir)
                elif kind == "credential":
                    # the API rejected the grader's credential (401 / 403). Not a
                    # defect of the run and not shedding: every grader of the family fails the same way until the credential is fixed. The FAMILY is
                    # held -- `dispatch.hold.<family>`, the operator's own file, which the gate re-reads before every later group of this tick -- ONE
                    # `problem` of kind `judge_auth` names the org and the mode (when this dispatch wrote the hold; a standing hold is reported by
                    # `held_family` as any operator hold), the copy is kept, nothing is quarantined, nothing counts toward RETRYABLE_CAP (the counter
                    # was cleared above); removing the file re-dispatches the family from the next tick
                    term = terminal_exception_line(tail)
                    org, named = credential_org_of_tail(tail)
                    mode = os.environ.get("SISTEER_SECRETS_MODE") or "op"
                    summary["credential"] = int(summary.get("credential", 0)) + len(rest)  # type: ignore[arg-type]
                    reason = (f"credential: the API rejected the {family} grader's credential (401 / 403) for org {org}"
                              f"{'' if named else ' (the default; the grader named none)'} under SISTEER_SECRETS_MODE={mode} -- fix the credential "
                              f"({'the federation ids file / the VM identity' if mode == 'wif' else 'ANTHROPIC_API_KEY' if mode == 'env' else 'the vault item'}) "
                              f"and remove this file; terminal: {term[:200]}; log {log_path}")
                    hold_file, written = family_hold_write(args.state_dir, family, reason, writer="credential")
                    log_event(args.state_dir, {"event": "dispatch_credential", "family": family, "model_key": model_key, "campaign": campaign, "n": len(rest),
                                               "run_ids": [str(r["run_id"]) for r in rest], "rc": rc, "dispatch_log": log_path, "terminal_exception": term,
                                               "org": org, "org_named_by_grader": named, "secrets_mode": mode, "hold_file": hold_file, "hold_written": written,
                                               "note": "the API rejected the grader's credential (401 / 403): the family is HELD (dispatch.hold.<family>) until the "
                                                       "operator fixes the credential and removes the file; the copy is kept, nothing is quarantined, nothing counts "
                                                       "toward the retryable cap"})
                    if written:
                        probs = summary.setdefault("problems", [])
                        if isinstance(probs, list):
                            probs.append(f"judge_auth: the API rejected the {family} grader's credential (401 / 403) for org {org} under SISTEER_SECRETS_MODE={mode}: "
                                         f"family held ({hold_file}); fix the credential and remove the file")
                        log_event(args.state_dir, {"event": "problem", "kind": "judge_auth", "family": family, "model_key": model_key, "campaign": campaign, "org": org,
                                                   "org_named_by_grader": named, "secrets_mode": mode, "n": len(rest), "run_ids": [str(r["run_id"]) for r in rest],
                                                   "rc": rc, "dispatch_log": log_path, "terminal_exception": term, "hold_file": hold_file,
                                                   "note": "ONE problem per credential event: every later group of the family is stopped at the hold gate this tick and "
                                                           "skipped on later ticks (held_family) until the file is removed; no per-run quarantine, no retries"})
                elif kind == "retryable":
                    summary["retryable"] = int(summary.get("retryable", 0)) + len(rest)  # type: ignore[arg-type]
                    log_event(args.state_dir, {"event": "dispatch_retryable", "family": family, "model_key": model_key, "campaign": campaign, "n": len(rest),
                                               "run_ids": [str(r["run_id"]) for r in rest],
                                               "rc": rc, "dispatch_log": log_path, "consecutive": n_retry, "cap": RETRYABLE_CAP, "terminal_exception": terminal_exception_line(tail),
                                               **({"filed": progress["filed"], "attempted": progress["attempted"], "unreached": progress["unreached"]} if progress else {}),
                                               "note": "upstream contention (the registry lock, or the live judge's bounded-retry shortfall): the copy is kept and the group is "
                                                       "dispatched again next tick; nothing quarantined until the cap -- charged per run to the attempts left incomplete "
                                                       "(a run filed this dispatch resets, a run never reached is not charged; the registry owner's ruling of 2026-10-13 23:29)"})
                else:
                    summary["failed"] += len(rest)
                    quarantine_add(args.state_dir, args.registry_root, group=rest, family=family, model_key=model_key, campaign=campaign, rc=rc, cmd=cmd, log_path=log_path)
    if refusals:                                                                     # A refusal is a problem every tick it stands, beside the counter
        fam0, _ids0, exc0 = refusals[0]
        n_ref = int(summary.get("refused", 0))
        probs = summary.setdefault("problems", [])
        if isinstance(probs, list):
            probs.append(f"dispatch_refused: {n_ref} run(s) refused: {exc0.reason}: {exc0.detail}")
        log_event(args.state_dir, {"event": "problem", "kind": "dispatch_refused", "n": n_ref, "reasons": sorted({e.reason for _f, _i, e in refusals}),
                                   "families": sorted({f for f, _i, _e in refusals}), "run_ids": sorted(rid for _f, ids, _e in refusals for rid in ids),
                                   "first": {"family": fam0, "reason": exc0.reason, "detail": exc0.detail, "config_path": exc0.path},
                                   "note": "pending run(s) the dispatch refused this tick (left pending, nothing claimed): fix the cause the first refusal names "
                                           "(a missing config: 05_stage_roster.sh --step configs; an unresolved placeholder: the store layout or the template)"})
    standing = {rid: e for rid, e in _NOT_FOLDED.items() if int(e.get("ticks") or 0) >= NOT_FOLDED_PROBLEM_TICKS}   # Not folded for N ticks
    if standing:
        worst = max(standing.values(), key=lambda e: int(e.get("ticks") or 0))
        probs = summary.setdefault("problems", [])
        if isinstance(probs, list):
            probs.append(f"dispatch_not_folded: {len(standing)} run(s) not folded for {int(worst.get('ticks') or 0)} tick(s): {worst.get('message')}")
        log_event(args.state_dir, {"event": "problem", "kind": "dispatch_not_folded", "n": len(standing), "run_ids": sorted(standing), "ticks": int(worst.get("ticks") or 0),
                                   "since_utc": min(str(e.get("since_utc")) for e in standing.values()), "message": worst.get("message"),
                                   "families": sorted({str(e.get("family")) for e in standing.values()}),
                                   "note": "these runs' claims have been refused as not folded on consecutive ticks (ClaimNotFolded: the run's or the judge's row has not "
                                           "folded): the harvester is not folding them -- read the harvester's cycle and the run's spool record; never quarantined, so the "
                                           "problem stands every tick until the fold lands or the run leaves the pending set"})
    standing_id = {rid: e for rid, e in _IDENTITY_CHANGED.items() if int(e.get("ticks") or 0) >= IDENTITY_CHANGED_PROBLEM_TICKS}   # refused on identity for N ticks
    if standing_id:
        worst = max(standing_id.values(), key=lambda e: int(e.get("ticks") or 0))
        probs = summary.setdefault("problems", [])
        if isinstance(probs, list):
            n_held = sum(1 for e in standing_id.values() if e.get("held"))
            probs.append(f"dispatch_identity_changed: {len(standing_id)} run(s) refused on identity for {int(worst.get('ticks') or 0)} tick(s)"
                         + (f" ({n_held} held: {DISPATCH_IDENTITY_HOLD_RELEASE} or a changed identity releases)" if n_held else "") + f": {worst.get('message')}")
        log_event(args.state_dir, {"event": "problem", "kind": "dispatch_identity_changed", "n": len(standing_id), "run_ids": sorted(standing_id), "ticks": int(worst.get("ticks") or 0),
                                   "held": sorted(rid for rid, e in standing_id.items() if e.get("held")),
                                   "since_utc": min(str(e.get("since_utc")) for e in standing_id.values()), "message": worst.get("message"),
                                   "families": sorted({str(e.get("family")) for e in standing_id.values()}),
                                   "note": "these runs' grader has refused on an identity change on consecutive ticks (the registry keeps moving under the binding, or a "
                                           "selected run keeps leaving the judged view): read the refusal text and the runs' currency; never quarantined, so the problem "
                                           "stands every tick until a dispatch ends otherwise or the runs leave the pending set"})
    # sweephold / sweephold-fix-1: the sweep hold file is read ONCE per tick, here, before the contrasts block --
    # the snapshot the sweep is handed, so a flip during the per-tick pass cannot make the tick report the opposite of what the sweep did -- and
    # said every tick while it stands (the way the operator hold is), whether or not the block runs this tick; the tick reports the snapshot
    # AND what the sweep actually did (`contrasts_sweep_hold.sweep`)
    sweep_hold_reason = contrasts_sweep_hold(args.state_dir)
    if sweep_hold_reason is not None:
        log_event(args.state_dir, {"event": "contrasts_sweep_held", "dry_run": bool(args.dry_run), "reason": sweep_hold_reason, "file": os.path.join(str(args.state_dir), CONTRASTS_SWEEP_HOLD),
                                   "note": "the once-per-advance contrast re-emission sweep is held by the file; remove it to release at the next tick"})
    operator_hold = dispatch_hold(args.state_dir)                                              # re-read before any emission
    if not args.contrasts:
        sweep_status = SWEEP_STATUS_DISABLED
    elif summary.get("hold") == "schema":
        sweep_status = SWEEP_STATUS_SCHEMA_HELD
    elif summary.get("held") or operator_hold is not None:
        sweep_status = SWEEP_STATUS_OPERATOR_HELD
    else:
        emit_contrasts_for(args.registry_root, args.model_id, args.campaign, args.spool, args.dry_run, args.state_dir)
        sweep_status = contrasts_sweep_for(args, summary, hold_reason=sweep_hold_reason)   # the campaign-wide re-emission sweep, once per daemon code advance
    summary["contrasts_sweep_hold"] = {"held": sweep_hold_reason is not None, "reason": sweep_hold_reason, "sweep": sweep_status}
    # under a schema stop seen mid-tick nothing more reads the registry (the prune asks it which runs are still pending)
    pr = mirror_prune_pass(args.registry_root, args.state_dir, args.dry_run, args.campaign) if summary.get("hold") != "schema" else {"dirs": 0, "bytes": 0}
    summary.setdefault("mirror_retired_dirs", 0); summary.setdefault("mirror_retired_bytes", 0)
    summary["mirror_pruned_dirs"] = pr["dirs"]; summary["mirror_pruned_bytes"] = pr["bytes"]
    summary["mirror_du_bytes"] = {sub: _du(root) for sub, root in sorted(mirror_roots.items()) if os.path.isdir(root)}
    summary["mirror_free_bytes"] = {sub: mirror_free_bytes(root) for sub, root in sorted(mirror_roots.items()) if os.path.isdir(root)}
    free_override = mirror_free_bytes_override()                # the test seam (MIRROR_FREE_BYTES_ENV), armed under pytest only: the figure every root above reported
    if free_override is not None:
        summary["mirror_free_bytes_override"] = free_override   # never a quiet substitution: the summary says the readings are stated, and what was stated
    free_stray = mirror_free_bytes_override_ignored()
    if free_stray is not None:                                  # a stray outside pytest: the real readings stood, and the summary + one event per tick name it
        summary["mirror_free_bytes_override_ignored"] = free_stray
        log_event(args.state_dir, {"event": "mirror_free_bytes_override_ignored", "variable": MIRROR_FREE_BYTES_ENV, "value": free_stray,
                                   "note": f"a test seam honoured only under pytest ({PYTEST_CURRENT_TEST_ENV} set); the real free-space readings stand -- unset it"})
    summary["mirror_prune_pending"] = len(_prune_ledger_load(args.state_dir))
    rows_now = rows_load(args.state_dir)                        # the ledger as this tick leaves it (the dispatches above may have opened items)
    summary["rows_pending"] = sum(1 for e in rows_now.values() if e.get("state") == ROWS_STATE_PENDING)
    summary["rows_exhausted"] = sum(1 for e in rows_now.values() if e.get("state") == ROWS_STATE_EXHAUSTED)
    if summary["rows_exhausted"]:                               # the named terminal state is a problem every tick it stands (never a silent drop)
        exhausted_ids = sorted(k for k, e in rows_now.items() if e.get("state") == ROWS_STATE_EXHAUSTED)
        probs = summary.setdefault("problems", [])
        if isinstance(probs, list):
            probs.append(f"rows_exhausted: {len(exhausted_ids)} run(s) with complete judgments and no derived rows after {ROWS_ATTEMPT_CAP} attempts "
                         f"(ledger {_rows_path(str(args.state_dir))}; re-arm with {DISPATCH_ROWS_RELEASE})")
        log_event(args.state_dir, {"event": "problem", "kind": "rows_exhausted", "n": len(exhausted_ids), "run_ids": exhausted_ids, "ledger": _rows_path(str(args.state_dir)),
                                   "note": f"judgments filed, derived rows missing after {ROWS_ATTEMPT_CAP} attempts: read the ledger's last_error, fix the cause, then re-arm with "
                                           f"`{DISPATCH_ROWS_RELEASE}` in the state dir"})
    log_event(args.state_dir, {"event": "tick", **summary})
    return summary


# ----------------------------------------------------------------------------------------------------- the Petri drive start
#: The daemon OWNS the Petri drive's start. The Petri proxy job serves the target on its cluster and writes
#: `<wave root>/petri/<tag>/serving/endpoint.json` when it is up; the drive
#: (scripts/sdfbase/full22/petri_drive_full22.sh <model> <view> <cluster> <wave sha>) runs on THIS host against that endpoint and files
#: every run into the registry spool. The Petri pass runs on its OWN cadence — a thread beside the grading loop, because a judge
#: subprocess may hold `tick` for hours while the proxy releases its GPU after 90 min without a driver heartbeat — and each
#: pass:
#:   * re-DISCOVERS the waves from the model's front-door state (`WAVE.<view>.json` + the petri unit's last `submitted` ledger row): the
#:     documented order starts this daemon before 06 writes the wave, so a startup-time list would never see it.
#:   * PREPARES each unit eagerly, while the proxy stands up: a PERSISTENT detached checkout of THE UNIT'S SHA — the sha its bound
#:     submission was made at (its submitted row's extra.sha / the attempts ledger, petri_unit_sha; never the wave file's, which a re-minted
#:     proxy or the overflow pass leaves behind) — under `<state-dir>/checkouts/<sha>` (the drive refuses a worktree whose HEAD is not
#:     the launched commit, and --repo moves on with every commit after 06), its `.venv` a link to the venv that owns the fit
#:     interpreter, the scenario bundle, the grading host's bank view of (model, view) built and verified at that sha (the drive refuses
#:     without it).
#:   * probes the endpoint (`test -f` + mtime + content, one ssh) and the LIVENESS of the proxy job the file names (`squeue -j`, a second):
#:     a stale endpoint.json left by a dead job never starts a drive.
#:   * keys everything on the ENDPOINT IDENTITY (the proxy job id, the endpoint's mtime, the unit's sha — the record's `wave_sha` field, kept
#:     by name): a completed drive is never started
#:     again for its identity; a failed one is retried on a later pass while the proxy is alive, up to PETRI_DRIVE_ATTEMPT_CAP attempts per
#:     identity; a replaced proxy is a new identity and a fresh count.
#:   * CLAIMS before it spawns, under a per-tag flock that covers EVERY read, reconcile and save of the tag's record: the
#:     attempt is written `claimed` before Popen and `started` after the child has published its pid; the child inherits the lock's fd and
#:     closes it only after writing that marker (`<log>.pid`: pid + /proc start time), so a daemon that dies between Popen and its
#:     `started` save leaves a lock no reconciler can take before the marker exists — the next pass ADOPTS the claim from the marker
#:     instead of declaring it failed and spawning a twin; a claim with no marker was never spawned and is `failed`.
#:   * SUPERVISES the child: its exit code is observed (Popen.poll while this process lives; the spawn wrapper writes `<log>.rc`, so a
#:     restarted daemon settles a drive it did not spawn) — and exit 0 is `completed` ONLY when the drive's result summary
#:     (`<log>.result.json`, written by petri_drive_full22.sh's verdict) says `complete` with every planned run verified FILED on this
#:     host's spool — and the ENDPOINT is complete only when the runs the drive filed cover the wave's EXPECTED run set, derived
#:     independently of any drive from the proxy job's own registration (`<shared root>/registry/pending/<job id>.json`, one run per
#:     (eval, object)) minus the objects the standup skipped by name. The daemon OWNS a drive of the FULL required scope:
#:     the child gets an ALLOWLIST of the daemon's environment — never F22_PETRI_CELLS or any other selection knob, which
#:     stay manual / dev knobs outside the daemon — so a drive that filed less than the endpoint requires is an ERROR: `partial:<covered>/
#:     <required>` is recorded (the coverage accounting is the truthful record), counts toward the cap like a failure, and after the cap
#:     `petri_drive_gave_up` names the uncovered runs; a completed full-scope drive touches the proxy's STOP, so nothing relaunches. Exit 3
#:     is `incomplete` (legs pending on overflow ledgers: the proxy released, a re-run pass a deliberate step, no retry); anything else
#:     `failed` with the log path.
#:   * PRUNES a checkout whose wave is CLOSED to the drive — no current wave names its sha any more (re-frozen at a newer sha, or gone), or
#:     every wave still at that sha has its drive `completed` — and nothing runs from it (`git worktree remove`; re-made on demand if a
#:     replaced proxy ever needs it again), so the state dir does not accumulate checkouts.
#: The record `<state-dir>/petri_drive.<tag>.json` = {tag, model, view, cluster, wave_sha, endpoint, attempts: [{n, state, identity, pid,
#: proc_start, log, rc_file, checkout, argv, claimed_utc, started_utc, exited_utc, exit_code, reason}]}; a human edits or deletes it to
#: override the cap. `--dry-run` prints the start it would make and records nothing.
#: No START under the operator's `dispatch.hold` or under the schema hold -- `petri_start_hold`, read right before the start
#: decision; settling and probing go on, nothing is claimed, prepared for a start or spawned, one `petri_drive_held` per tag (said again after
#: a hold lifts and returns).
PETRI_DRIVE_SCRIPT_DEFAULT = "scripts/sdfbase/full22/petri_drive_full22.sh"
PETRI_SPEC_KEYS = ("tag", "model", "view", "cluster", "ssh", "endpoint", "sha")
#: THE OVERFLOW PASS (petri-completion, 2026-10-13): a spec carries `pass` = `first` (absent = first: every spec before this field) or
#: `overflow` — the CONTEXT-OVERFLOW RE-RUN proxy 06 --pass overflow submitted (unit `petri.overflow`, endpoint `<wave root>/petri/<tag>/
#: serving_rerun/endpoint.json`), with its own record and lock under tag `<tag>.overflow` (`base_tag` keeps the wave's) and
#: `first_job_id` = the first-pass spec's bound job. Its drive is `petri_drive_full22.sh <model> <view> <cluster> <sha> ledgered` under
#: F22_OVERFLOW_RERUN=1 — the daemon's OWN setting from the spec, applied AFTER the deny-list stripped any inherited copy (the operator still
#: cannot select legs or passes; the drive reads the ledgered legs itself through petri_run.py legs-status) — and it STARTS only once the
#: first pass's record holds an attempt for `first_job_id` settled `incomplete` and none live (`petri_overflow_waiting_first_pass` until then).
PETRI_PASS_FIRST, PETRI_PASS_OVERFLOW = "first", "overflow"
PETRI_OVERFLOW_UNIT = "petri.overflow"
PETRI_OVERFLOW_LEGS_ARG = "ledgered"
PETRI_OVERFLOW_ENV = "F22_OVERFLOW_RERUN"
PETRI_OVERFLOW_RECORD_SUFFIX = ".overflow"
#: the drive's result `handoff.verify_scope` values (petri_drive_full22.sh): `all` / `selected` verified through petri_registry_handoff.py
#: verify; `control_only` = every selected leg is a control leg (no registered run in scope: verify not run, completion is the drive's
#: verdict over `legs.detail`)
PETRI_VERIFY_SCOPE_CONTROL_ONLY = "control_only"
#: a drive that exits non-zero is retried on later passes up to this many attempts PER ENDPOINT IDENTITY, then `petri_drive_gave_up`
PETRI_DRIVE_ATTEMPT_CAP = 3
#: the drive's own constants (petri_drive_full22.sh): the frozen scenario bundle, the grading host bank-view root, the roster it checks against.
#: THE BANK ROOT IS HANDED TO THE DRIVE (demo-import, 2026-10-11): `--petri-bank-root` (default: the paper's legacy the grading host bank, the drive's
#: own default) is where this daemon BUILDS the view before a start AND what the spawned drive READS — the pass exports it as
#: F22_GRADING_HOST_BANK_ROOT (the drive's override of its literal). Before this, 08_grade.sh passed no root and the drive hardcoded the legacy
#: one, so a front-door model staged into $SISTEER_OBJECTS_BANK was prepared against a bank the demo never wrote (`petri_drive_unprepared`:
#: no object.pt) while the drive itself would have read a third place. 08_grade.sh now passes --petri-bank-root "$SISTEER_OBJECTS_BANK".
PETRI_BUNDLE_DIR_DEFAULT = os.path.expanduser("~/.local/state/petri/scenarios/8812ba083bb5473c")
#: the bank-view root: under the site's staging directory ($STAGE, src/sisteer/site.py), expanded on this host; left as written where
#: the site names no staging directory, and petri_prepare then refuses it by name before anything is built under it
PETRI_BANK_ROOT_PLACEHOLDER = "$STAGE/sdfbase/full22_objects"


def _stage_bank_root() -> str:
    try:
        from sisteer import site
        return str(site.expand_path(PETRI_BANK_ROOT_PLACEHOLDER))
    except (ImportError, ValueError):     # site.PlaceholderError / site.SiteConfigError: no staging directory on this host
        return PETRI_BANK_ROOT_PLACEHOLDER


PETRI_GRADING_HOST_BANK_ROOT_DEFAULT = _stage_bank_root()
PETRI_ROSTER_DEFAULT = "configs/sdfbase_full22_roster.yaml"
#: what the fit interpreter must import for the drive (petri_drive_full22.sh checks the same two)
PETRI_IMPORT_CHECK = "import inspect_ai, petri_bloom"
#: the Petri pass's own cadence (seconds) and its floor; the grading loop's floor (`--interval`) is TICK_INTERVAL_FLOOR_S
PETRI_INTERVAL_DEFAULT_S = 60.0
PETRI_INTERVAL_FLOOR_S = 5.0
TICK_INTERVAL_FLOOR_S = 30.0
PETRI_JOIN_TIMEOUT_S = 600.0
#: the spawn wrapper: publishes ITS pid + /proc start time to $PETRI_DRIVE_PID_FILE first (atomically), closes the claim lock's inherited fd
#: ($PETRI_LOCK_FD) only then, runs the drive, and leaves the exit code in $PETRI_DRIVE_RC_FILE — so a daemon that did not spawn it can
#: adopt and settle it, and no reconciler can take the tag's lock before the marker exists
PETRI_WRAPPER = ('st=$(sed "s/.*) //" /proc/$$/stat | awk \'{print $20}\'); '
                 'printf \'{"pid": %d, "proc_start": %s, "utc": "%s"}\\n\' "$$" "${st:-null}" "$(date -u +%FT%TZ)" > "${PETRI_DRIVE_PID_FILE:?}.tmp" '
                 '&& mv -f "${PETRI_DRIVE_PID_FILE}.tmp" "$PETRI_DRIVE_PID_FILE"; '
                 '[ -z "${PETRI_LOCK_FD:-}" ] || eval "exec $PETRI_LOCK_FD>&-"; '
                 'bash "$0" "$@"; rc=$?; printf "%s\\n" "$rc" > "${PETRI_DRIVE_RC_FILE:?}"; exit "$rc"')
#: the drive's "legs INCOMPLETE on overflow ledgers, none failed" exit code (petri_drive_full22.sh f22_drive_verdict): not a success, not a failure
PETRI_DRIVE_RC_OVERFLOW = 3
#: the drive's "a previous attempt's legs still hold the per-leg locks" refusal (petri_drive_full22.sh, before any leg is launched): `blocked`,
#: not counted toward the cap; the daemon stops that attempt's surviving session before the next start
PETRI_DRIVE_RC_LOCKS_HELD = 4
#: THE DRIVE'S ENVIRONMENT: an ALLOWLIST of the daemon's — exact names and prefixes — never a selection knob (the deny list wins over
#: everything, `--petri-drive-env-allow` included): the daemon's drive is always the full required scope. Secrets the drive has no use for
#: (GH / HF-cluster / Modal / RunPod / W&B tokens) are simply not on the list; what the drive and its children read is (petri_drive_full22.sh,
#: petri_run.py, the registry emitters, sisteer.vault under op mode: the SISTEER_OP_ITEM_* names, the site's SISTEER_* / CLUSTER__* block, the
#: run-overrides layer, ssh's agent socket, the OpenRouter key for the auditor chain).
PETRI_DRIVE_ENV_DENY = ("F22_PETRI_CELLS", "F22_PETRI_OBJECTS_ONLY", "F22_PETRI_RELEASE_SCOPE", "F22_OVERFLOW_RERUN", "F22_SECOND_PASS")
#: The operator OVERLAY for the drive's environment: `<state-dir>/petri_drive_env.json` ({NAME: value}) is read at EVERY pass and merged over
#: the allowlisted base, so a knob the drive reads reaches the NEXT attempt without a wave re-freeze and without restarting the unit. The
#: overlay is EXACTLY three knobs, each validated (an overlay that rode the whole daemon-env allowlist could set SISTEER_SECRETS_MODE /
#: PATH / HOME / OP_*, under which the drive skips the vault and uses an inherited key while still logging the org it was asked for):
#: F22_AUDITOR_ORG, a vault item the drive accepts (`batch` is the Batch-API-only pool, never live concurrents, and the drive refuses it
#: by name); PETRI_AUDITOR_RETRY_SLEEP_CAP_S, a positive integer of seconds up to an hour; PETRI_EVAL_MAX_CONNECTIONS, a positive integer
#: up to the configured ceiling on the auditor's concurrency. A PRESENT but invalid file (unreadable, not JSON, not an object, a name
#: outside the three, a value outside its range) REFUSES the attempt by name before anything is recorded or spawned, never a partial
#: application (a truncated file or a misspelled F22_AUDITOR_ORG would otherwise launch the next attempt on the default pool and repeat
#: the stall against an idle Petri GPU); the refusal stands until the file is fixed or removed. An absent file is no overlay. The applied
#: NAMES are logged, never a value. The pool the drive then uses is PROVEN, not asserted: the daemon hashes the named org's vault item in
#: a shell pipeline (`petri_expected_key_sha12`, the value never enters this process), the drive's DRIVE_START records the hash of the key
#: it resolved, and `_petri_attest_key` compares the two.
PETRI_DRIVE_ENV_FILE = "petri_drive_env.json"
PETRI_DRIVE_OVERLAY_ORGS = ("lowprio", "highprio")
PETRI_DRIVE_OVERLAY_RETRY_CAP_MAX_S = 3600
PETRI_DRIVE_OVERLAY_WIDTH_MAX = 100
PETRI_DRIVE_OVERLAY_KNOBS = ("F22_AUDITOR_ORG", "PETRI_AUDITOR_RETRY_SLEEP_CAP_S", "PETRI_EVAL_MAX_CONNECTIONS")
PETRI_DRIVE_ENV_ALLOW = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TERM", "TMPDIR", "TZ", "SISTEER_SECRETS_MODE", "OPENROUTER_API_KEY", "F22_AUDITOR_ORG",
                         "ANTHROPIC_API_KEY", "HF_TOKEN", "F22_ROSTER", "F22_SERVING_PATH", "F22_AUDITOR_ORG", "F22_REGISTRY_HANDOFF", "F22_PETRI_TARGET_SEED_BASE")
PETRI_DRIVE_ENV_ALLOW_PREFIXES = ("SISTEER_", "CLUSTER__", "OVERRIDE_", "OP_", "SSH_", "LC_", "XDG_", "PETRI_", "INSPECT_", "HF_", "ROMP_")
#: under SISTEER_SECRETS_MODE=wif these never reach the drive (petri_drive_env): a static credential would shadow federation for every leg
PETRI_DRIVE_WIF_SCRUB = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class OverlayRefused(ValueError):
    """A present but invalid operator overlay: `file` and the `reasons` (names and shapes only, never a value)."""

    def __init__(self, file: str, reasons: Sequence[str]):
        super().__init__(f"{file}: " + "; ".join(reasons))
        self.file = file
        self.reasons = list(reasons)


def _overlay_positive_int(name: str, value: object, hi: int) -> Tuple[Optional[str], Optional[str]]:
    """(the knob as a string, None) for a positive integer <= hi (an int, or a string of digits), else (None, the reason, type only)."""
    if isinstance(value, bool) or not (isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit())):
        return None, f"{name}: a positive integer <= {hi} (got a {type(value).__name__})"
    n = int(value)
    if not 1 <= n <= hi:
        return None, f"{name}: a positive integer <= {hi} (got {n})"
    return str(n), None


def petri_drive_env_overlay(state_dir: Optional[str]) -> Tuple[Dict[str, str], bool]:
    """The operator overlay (PETRI_DRIVE_ENV_FILE) -> (the knobs to apply, whether a file was present). Raises OverlayRefused for a present
    but invalid file: the whole file, never the valid part of it."""
    if not state_dir:
        return {}, False
    path = os.path.join(state_dir, PETRI_DRIVE_ENV_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        return {}, False
    except OSError as exc:
        raise OverlayRefused(path, [f"unreadable: {type(exc).__name__}"]) from exc
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise OverlayRefused(path, [f"not JSON: {exc}"]) from exc
    if not isinstance(doc, dict):
        raise OverlayRefused(path, ["not a JSON object of {NAME: value}"])
    applied: Dict[str, str] = {}
    reasons: List[str] = []
    for k, v in doc.items():
        k = str(k)
        if k not in PETRI_DRIVE_OVERLAY_KNOBS:
            reasons.append(f"{k}: not an overlay knob (the overlay is exactly {', '.join(PETRI_DRIVE_OVERLAY_KNOBS)})")
        elif k == "F22_AUDITOR_ORG":
            if isinstance(v, str) and v in PETRI_DRIVE_OVERLAY_ORGS:
                applied[k] = v
            else:
                reasons.append(f"F22_AUDITOR_ORG: one of {', '.join(PETRI_DRIVE_OVERLAY_ORGS)} (got a {type(v).__name__} that is neither)")
        else:
            val, why = _overlay_positive_int(k, v, PETRI_DRIVE_OVERLAY_RETRY_CAP_MAX_S if k == "PETRI_AUDITOR_RETRY_SLEEP_CAP_S" else PETRI_DRIVE_OVERLAY_WIDTH_MAX)
            if why:
                reasons.append(why)
            else:
                applied[k] = val  # type: ignore[assignment]
    if reasons:
        raise OverlayRefused(path, reasons)
    return applied, True


def petri_drive_env(base: Mapping[str, str], extra_allow: Sequence[str] = (), state_dir: Optional[str] = None) -> Tuple[Dict[str, str], List[str], List[str]]:
    """The drive's environment from the daemon's: (the allowlisted variables with the validated overlay applied, the selection knobs that were
    present and DROPPED, the overlay NAMES applied). `extra_allow` adds exact names or `PREFIX*` entries (`--petri-drive-env-allow`); the
    deny list wins over every allow; the overlay never rides the allowlist (it is exactly PETRI_DRIVE_OVERLAY_KNOBS). Raises OverlayRefused."""
    names = set(PETRI_DRIVE_ENV_ALLOW) | {x for x in extra_allow if not x.endswith("*")}
    prefixes = tuple(PETRI_DRIVE_ENV_ALLOW_PREFIXES) + tuple(x[:-1] for x in extra_allow if x.endswith("*") and len(x) > 1)
    env: Dict[str, str] = {}; dropped: List[str] = []
    for k, v in base.items():
        if k in PETRI_DRIVE_ENV_DENY:
            dropped.append(k)
        elif k in names or k.startswith(prefixes):
            env[k] = v
    env.setdefault("SISTEER_SECRETS_MODE", "op")
    if env["SISTEER_SECRETS_MODE"] == "wif":
        # under federation NO Anthropic key reaches the drive -- a present one (allowlisted for env mode)
        # would shadow federation for every leg's SDK, and the drive refuses it by name before its first leg; dropped here (named in
        # `dropped`, never a value), as reproduce/lib/secrets.sh's secrets_unit_key_line scrubs it from a generated unit under the same mode
        for k in PETRI_DRIVE_WIF_SCRUB:
            if env.pop(k, None) is not None:
                dropped.append(k)
    overlay, present = petri_drive_env_overlay(state_dir)
    if present:
        log_event(state_dir, {"event": "petri_drive_env_overlay", "applied": sorted(overlay), "file": os.path.join(state_dir or "", PETRI_DRIVE_ENV_FILE),
                              "note": "operator knobs for the drive's next attempt (names only; exactly the three knobs, validated; an invalid file refuses the attempt)"})
        env.update(overlay)
    return env, sorted(dropped), sorted(overlay)


def _vault_item_sha12(env: Mapping[str, str], org: str, run=subprocess.run) -> Tuple[Optional[str], Optional[str]]:
    """sha256[:12] of the named org's vault item, hashed in a shell PIPELINE so the value never enters this process; (None, reason) when
    the fetch fails. The pipeline lives in sisteer.vault.secret_sha12 -- the ONE sanctioned caller of the secret-manager CLI."""
    from sisteer.vault import secret_sha12
    return secret_sha12(org, env=env, run=run, timeout=60.0)


def petri_expected_key_sha12(env: Mapping[str, str], run=subprocess.run, *, switched: bool = False) -> Dict[str, object]:
    """What the drive's ONE Anthropic key SHOULD hash to (sha256[:12]), so the pool its DRIVE_START attests (`auditor_key_sha12=`) is proven,
    not asserted. Under op mode: the named org's vault item (`_vault_item_sha12`). Under SISTEER_SECRETS_MODE=env the drive uses the inherited
    ANTHROPIC_API_KEY whatever the org label says, so the label is proven INDEPENDENTLY (hashing the same inherited key on both sides attested
    a lowprio key as highprio): the in-process key's hash must equal the named vault item's, fetched through the pipeline; a mismatch REFUSES
    (`mismatch`), and a vault that cannot be reached refuses an org the overlay SWITCHED to (`switched`) while a default org is attested
    unverified (`verified: False`, recorded). Returns {mode, org, sha12 | None, verified, error | None, kind, auth}. Under
    SISTEER_SECRETS_MODE=wif there is NO key to hash: the drive attests the digest of the org's federation ids FILE
    (`auditor_auth=wif:<org> ids_sha12=`) and the daemon expects sha256(<that file>)[:12] through the vault
    (`sisteer.vault.wif_ids_sha12`, the ONE reader of the workload-identity env file), resolved under the DRIVE's environment (its
    SISTEER_WIF_ENV_DIR / HOME); `auth: "wif"`, `verified: True` (the daemon read the file itself); an absent, empty or incomplete file is
    `unresolved` by name and no attempt starts (the drive would die on the same read)."""
    mode = env.get("SISTEER_SECRETS_MODE") or "op"
    org = env.get("F22_AUDITOR_ORG") or "lowprio"
    out: Dict[str, object] = {"mode": mode, "org": org, "sha12": None, "verified": False, "error": None, "kind": None, "auth": "key"}
    if mode == "wif":
        from sisteer.vault import PoolUnavailable, wif_env_path, wif_ids_sha12
        out["auth"] = "wif"
        try:
            out.update({"sha12": wif_ids_sha12(org, env=env), "verified": True, "ids_file": os.path.basename(wif_env_path(org, env=env))})
        except (PoolUnavailable, ValueError, OSError) as exc:
            out.update({"error": f"SISTEER_SECRETS_MODE=wif: {exc}", "kind": "unresolved"})
        return out
    if mode != "env":
        vault, why = _vault_item_sha12(env, org, run)
        out.update({"sha12": vault, "verified": vault is not None, "error": why, "kind": None if vault else "unresolved"})
        return out
    key = env.get("ANTHROPIC_API_KEY") or ""
    if not key:
        out.update({"error": "SISTEER_SECRETS_MODE=env but the drive's environment carries no ANTHROPIC_API_KEY", "kind": "unresolved"})
        return out
    inproc = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    vault, why = _vault_item_sha12(env, org, run)
    out["inproc_sha12"] = inproc; out["vault_sha12"] = vault
    if vault is not None and vault != inproc:
        out.update({"error": f"SISTEER_SECRETS_MODE=env: the inherited ANTHROPIC_API_KEY ({inproc}) is not the vault item {org!r} ({vault}): the org label would be false",
                    "kind": "mismatch"})
        return out
    if vault is None and switched:
        out.update({"error": f"SISTEER_SECRETS_MODE=env: the overlay names org {org!r} but the vault item cannot be fetched to prove the inherited key is it ({why})",
                    "kind": "unresolved"})
        return out
    out.update({"sha12": inproc, "verified": vault is not None, "error": None if vault is not None else f"unverified against the vault ({why})"})
    return out


def _petri_attest_key(att: dict, state_dir: Optional[str], tag: Optional[str]) -> bool:
    """The drive's DRIVE_START line attests the hash of the key it RESOLVED (`auditor_key_sha12=`): compared ONCE with the hash the daemon expected
    for the org it asked for: proven, not asserted. A drive that predates the field is recorded `unattested` once. Returns whether
    the record changed."""
    if att.get("key_attested") is not None or not att.get("log"):
        return False
    try:
        with open(att["log"], encoding="utf-8", errors="replace") as fh:
            line = next((ln for ln in fh if "DRIVE_START " in ln), None)
    except OSError:
        return False
    if line is None:
        return False
    exp = att.get("auditor") or {}
    exp_auth = exp.get("auth") or "key"
    m = re.search(r"\bauditor_key_sha12=([0-9a-f]{12})\b", line)
    mw = re.search(r"\bauditor_auth=wif:(\w+)\s+ids_sha12=([0-9a-f]{12})\b", line)      # the federated drive's attestation
    mo = re.search(r"\bauditor_org=(\w+)", line)
    org_logged = mo.group(1) if mo else None
    if m is None and mw is None:
        att["key_attested"] = "unattested"
        att["attestation_note"] = ("the drive at this wave sha records no auditor_key_sha12 / auditor_auth=wif:<org> ids_sha12= in DRIVE_START: the pool is asserted, "
                                   "not proven; re-freeze the wave at a sha carrying the attestation")
        log_event(state_dir, {"event": "petri_drive_key_unattested", "tag": tag, "attempt": att.get("n"), "org_expected": exp.get("org"), "org_logged": org_logged,
                              "auth_expected": exp_auth, "note": att["attestation_note"]
                              + (" (under SISTEER_SECRETS_MODE=wif this REFUSES the attempt: stopped while it runs, never completed, never retried)" if exp_auth == "wif" else "")})
        return True
    # the attestation of the auth the daemon expected: a key drive attests the key's hash, a federated drive the ids file's digest and the org it
    # federated as; a drive attesting the OTHER auth (a wave sha on the other path) can never match -- the record says which was logged
    auth_logged = "wif" if (mw is not None and (exp_auth == "wif" or m is None)) else "key"
    sha12_logged = mw.group(2) if auth_logged == "wif" else m.group(1)
    org_attested = mw.group(1) if auth_logged == "wif" else None
    ok = (bool(exp.get("sha12")) and auth_logged == exp_auth and sha12_logged == exp.get("sha12")
          and (org_logged is None or org_logged == exp.get("org")) and (org_attested is None or org_attested == exp.get("org")))
    att["key_attested"] = ok
    log_event(state_dir, {"event": "petri_drive_key_attested", "tag": tag, "attempt": att.get("n"), "match": ok, "org_expected": exp.get("org"), "org_logged": org_logged,
                          "auth_expected": exp_auth, "auth_logged": auth_logged, "org_attested": org_attested,
                          "sha12_logged": sha12_logged, "sha12_expected": exp.get("sha12")})
    if not ok:
        if auth_logged != exp_auth:
            why = (f"the drive attested {'a key' if auth_logged == 'key' else 'federation'} while the daemon runs under SISTEER_SECRETS_MODE={exp.get('mode')}: "
                   "the drive at this wave sha runs the other auth path (re-freeze the wave at a sha carrying this mode's attestation)")
        elif exp_auth == "wif":
            why = ("the federation ids the drive attested are not the named org's ids file as the daemon read it: the attested pool is false (another org's "
                   "file, a changed or forged ids file, or a drive-side override)")
        else:
            why = "the key the drive resolved is not the named org's vault item: the attested pool is false (an inherited key, a stale vault item, or a drive-side override)"
        att["attestation_note"] = why
        log_event(state_dir, {"event": "problem", "kind": "petri_auditor_key_mismatch", "tag": tag, "attempt": att.get("n"), "org_expected": exp.get("org"), "org_logged": org_logged,
                              "auth_expected": exp_auth, "auth_logged": auth_logged,
                              "note": why + (" -- under SISTEER_SECRETS_MODE=wif this REFUSES the attempt: stopped while it runs, never completed, never retried"
                                             if exp_auth == "wif" else "")})
    return True


#: the attempt state of a federated drive whose attestation was refused: terminal like `incomplete` (never retried for its endpoint), never a success
PETRI_ATTEMPT_UNATTESTED = "unattested"


def _petri_attestation_refusal(att: Mapping[str, object], *, settling: bool = False) -> Optional[str]:
    """Under SISTEER_SECRETS_MODE=wif an attempt whose DRIVE_START attestation is ABSENT (`unattested`: a wave sha from before the field), a
    KEY'S (a wave sha still on the key path -- an inherited key of any org) or MISMATCHED (another org's ids file, a rotated or forged one, a
    drive-side override) is REFUSED: its drive is stopped while it runs (so nothing is filed under an unproven pool), the attempt settles
    `unattested` -- never `completed`, never `partial`, the endpoint's coverage untouched -- and it is never retried for this endpoint
    identity (a retry at the same wave sha would attest the same). Returns the reason, or None: the attestation matched, has not been read
    yet (`key_attested` None while the drive runs -- its DRIVE_START line not written yet; at SETTLEMENT, `settling`, a never-read
    attestation is ABSENT and refused: an exit with no DRIVE_START line in the log proves no pool), or the attempt is a KEY-mode one (op /
    env stay exactly as they were: a mismatch there is a `problem` beside the result)."""
    if ((att.get("auditor") or {}).get("auth") or "key") != "wif":
        return None
    ka = att.get("key_attested")
    if ka is True or (ka is None and not settling):
        return None
    remedy = ("never completed, never retried for this endpoint: re-freeze the wave at a sha carrying this mode's attestation (or fix the named org's "
              "ids file), then edit or delete the record")
    if ka is None:
        return (f"attestation refused under SISTEER_SECRETS_MODE=wif: the attempt's log carries no DRIVE_START line at all (the drive exited before it "
                f"attested anything) -- no pool is proven; {remedy}")
    if ka == "unattested":
        return (f"attestation refused under SISTEER_SECRETS_MODE=wif: DRIVE_START carries no auditor_auth=wif:<org> ids_sha12= (the drive at this wave sha "
                f"predates the federated attestation or runs the key path) -- the pool is asserted, not proven; {remedy}")
    return f"attestation refused under SISTEER_SECRETS_MODE=wif: {att.get('attestation_note') or 'the attested federation identity is not the one expected'}; {remedy}"


def _petri_stop_unattested(att: dict, state_dir: Optional[str], tag: Optional[str]) -> bool:
    """A RUNNING attempt whose attestation was refused (:func:`_petri_attestation_refusal`) is stopped at once -- TERM to its session's process
    group (the wrapper's pid, start_new_session) -- so its legs file nothing under a pool the daemon cannot prove; recorded once on the attempt
    (`attestation_stop`) and as a `petri_drive_stopped_unattested` event. The settlement then reads the refusal and states the attempt `unattested`
    (:func:`_petri_reconcile`); members that outlive the wrapper are the settled attempt's survivors (`_petri_stop_survivors`). Returns whether the
    record changed."""
    if att.get("attestation_stop") is not None or not isinstance(att.get("pid"), int):
        return False
    why = _petri_attestation_refusal(att)
    if why is None:
        return False
    try:
        os.killpg(att["pid"], signal.SIGTERM)
        delivered = True
    except (ProcessLookupError, PermissionError):
        delivered = False
    att["attestation_stop"] = {"utc": utc_now(), "pid": att["pid"], "signal": "SIGTERM", "delivered": delivered, "reason": why}
    log_event(state_dir, {"event": "petri_drive_stopped_unattested", "tag": tag, "attempt": att.get("n"), "pid": att["pid"], "delivered": delivered,
                          "key_attested": att.get("key_attested"), "org_expected": (att.get("auditor") or {}).get("org"), "reason": why,
                          "note": "the running drive's attestation was refused under SISTEER_SECRETS_MODE=wif: TERM sent to its session so no leg files under an "
                                  "unproven pool; the attempt settles `unattested` (never completed, never retried for this endpoint)"})
    return True


#: after a settled attempt's survivors were sent TERM, how long before they are sent KILL (a leg that ignores TERM holds its lock)
SURVIVOR_KILL_AFTER_S = 60.0


def _boot_epoch() -> Optional[float]:
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("btime "):
                    return float(ln.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _utc_epoch(utc: Optional[str]) -> Optional[float]:
    """Epoch seconds of one of this module's `utc_now()` stamps (None for an absent or unreadable one)."""
    if not utc:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.strptime(str(utc)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _session_members(leader: int, started_after: Optional[float] = None, started_before: Optional[float] = None) -> List[int]:
    """Live (non-zombie) processes whose SESSION id is `leader`, the leader itself excluded. The drive is spawned with start_new_session, so
    its wrapper's pid is the session (and process group) of every leg, heartbeat and tunnel it starts, and a session id is never reused
    while a member lives: this names an attempt's survivors without trusting a pid or an argv. (2026-10-01: a SIGTERM to the attempt-3 wrapper
    alone left its four legs auditing under the old configuration and holding the per-leg locks; attempts 4-6 each read `already locked,
    skip` on every leg as 4 failed and the cap was burnt in four minutes.) Against a pid REUSED by an unrelated session leader after the
    attempt died, a member counts only when its own start time (/proc starttime over the boot epoch) falls inside the attempt's life
    [`started_after`, `started_before` + a minute]: a drive's legs are started while the drive runs, never after it exited."""
    out: List[int] = []
    boot = _boot_epoch(); tck = float(os.sysconf("SC_CLK_TCK") or 100)
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == leader:
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="utf-8") as fh:
                rest = fh.read().rsplit(")", 1)[1].split()
            if int(rest[3]) != leader or rest[0] == "Z":
                continue
            if boot is not None and (started_after is not None or started_before is not None):
                t0 = boot + int(rest[19]) / tck
                if (started_after is not None and t0 < started_after - 60.0) or (started_before is not None and t0 > started_before + 60.0):
                    continue
            out.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(out)


def _petri_stop_survivors(att: dict, state_dir: Optional[str], tag: Optional[str]) -> List[int]:
    """A SETTLED attempt's surviving session members get TERM once (recorded in the attempt as `survivors`), then KILL once SURVIVOR_KILL_AFTER_S
    has passed since the TERM; the whole process group (the leader's pid, start_new_session) and each member by pid. Returns the members still
    alive at this call: non-empty means the per-leg locks may still be held and no new attempt may start."""
    pid = att.get("pid")
    if not isinstance(pid, int):
        return []
    members = _session_members(pid, _utc_epoch(att.get("started_utc") or att.get("claimed_utc")), _utc_epoch(att.get("exited_utc")))
    if not members:
        return []
    surv = att.setdefault("survivors", {})
    now = time.time()
    sig = None
    if surv.get("term_t") is None:
        sig, surv["term_t"], surv["term_utc"] = signal.SIGTERM, now, utc_now()
    elif now - float(surv["term_t"]) >= SURVIVOR_KILL_AFTER_S and surv.get("kill_t") is None:
        sig, surv["kill_t"], surv["kill_utc"] = signal.SIGKILL, now, utc_now()
    surv["pids"] = members
    if sig is not None:
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        for m in members:
            try:
                os.kill(m, sig)
            except (ProcessLookupError, PermissionError):
                pass
        log_event(state_dir, {"event": "petri_drive_legs_stopped", "tag": tag, "attempt": att.get("n"), "pid": pid, "pids": members, "signal": sig.name,
                              "note": "the attempt exited but members of its session (legs, heartbeat, tunnel) were alive: they hold the per-leg locks, "
                                      "so no new attempt starts until they are gone" + ("" if sig == signal.SIGTERM else f" (TERM ignored for {SURVIVOR_KILL_AFTER_S:g}s)")})
    return members


def _petri_survivors_block(rec: dict, state_dir: Optional[str], tag: Optional[str]) -> List[int]:
    """Every settled attempt's survivors (stopped as above): the pids still alive, union over the record. Non-empty blocks a start."""
    alive: set = set()
    for a in rec.get("attempts") or []:
        if isinstance(a.get("pid"), int) and a.get("state") not in ("claimed", "started"):
            alive.update(_petri_stop_survivors(a, state_dir, tag))
    return sorted(alive)
#: how long the spawner waits for the wrapper's pid marker (it is the wrapper's first action) before recording the start from Popen alone
PETRI_MARKER_WAIT_S = 60.0


def parse_petri_clusters(specs: Optional[Sequence[str]]) -> Dict[str, Dict[str, str]]:
    """`--petri-cluster <label>=<ssh alias>=<that cluster's wave root>[=<its shared root>]` (repeatable) → label -> {ssh, wave_root, shared};
    the shared root locates the proxy job's registration (`<shared>/registry/pending/<job>.json`); absent, it is the wave root's `$SHARED`
    (the part before `/exp/`). Refused by name otherwise."""
    out: Dict[str, Dict[str, str]] = {}
    for spec in specs or ():
        parts = str(spec).split("=", 3)
        if len(parts) not in (3, 4) or not all(x.strip() for x in parts):
            raise ValueError(f"--petri-cluster {spec!r}: expected <label>=<ssh alias>=<wave root>[=<shared root>]")
        wave_root = parts[2].strip().rstrip("/")
        shared = parts[3].strip().rstrip("/") if len(parts) == 4 else _shared_of(wave_root)
        out[parts[0].strip()] = {"ssh": parts[1].strip(), "wave_root": wave_root, "shared": shared}
    return out


def _shared_of(path: str) -> str:
    """A cluster path's `$SHARED`: the part before `/exp/` (the wave root and the endpoint both live under `$SHARED/exp/`)."""
    return path.split("/exp/", 1)[0] if "/exp/" in path else ""


_PETRI_DISCOVERY_SAID: set = set()
#: the funnel's two ledgers under `<state root>` (reproduce/lib/submit_twin.py): the submissions rows (`submitted`, with `extra.sha` = the sha
#: the job was SUBMITTED at) and the attempts ledger (`attempt`, the FULL spec the funnel launched, `sha` included)
PETRI_SUBMISSIONS_LEDGER, PETRI_ATTEMPTS_LEDGER = "submissions.jsonl", "attempts.jsonl"


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def petri_unit_sha(row: Mapping[str, object], attempts: Sequence[Mapping[str, object]]) -> Tuple[Optional[str], str]:
    """THE SHA A PETRI UNIT'S DRIVE CHECKS OUT (petri-completion): the sha that unit's BOUND SUBMISSION was
    made at — the `submitted` row's `extra.sha` (submit_twin records it on every row), cross-checked against the funnel's attempts ledger (the
    `attempt` event of the same attempt id, whose `sha` / `spec.sha` is the spec the funnel launched); the ledger alone when an older row has no
    `extra.sha`. Returns (sha, where it was read) or (None, why not): no sha on either record, or the two disagreeing, is a refusal by name —
    NEVER the wave file's sha. Before this the drive ran at `WAVE.<view>.json`'s wave_sha, which `06 --pass overflow` deliberately never rewrites and
    a `--resubmit` / `--units petri.proxy` re-mint leaves unchanged: on the smoke the battery WAVE pinned a sha whose drive had neither `legs=ledgered`
    nor `petri_run.py legs-status`, so the overflow drive would have run incompatible code against the proxy's tree, and a re-minted first-pass
    proxy's old drive could not write the per-leg result the overflow preflight reads."""
    row_sha = str((row.get("extra") or {}).get("sha") or "") if isinstance(row.get("extra"), Mapping) else ""
    aid, job, lwid = str(row.get("attempt_id") or ""), str(row.get("job_id") or ""), str(row.get("logical_work_id") or "")
    same = [a for a in attempts if a.get("event") == "attempt"
            and ((aid and str(a.get("attempt_id") or "") == aid) or (not aid and job and str(a.get("job_id") or "") == job and str(a.get("logical_work_id") or "") == lwid))]
    att_sha = ""
    if same:
        a = same[-1]
        att_sha = str(a.get("sha") or ((a.get("spec") or {}).get("sha") if isinstance(a.get("spec"), Mapping) else "") or "")
    if row_sha and att_sha and row_sha != att_sha:
        return None, (f"the submissions row's extra.sha {row_sha} and the attempts ledger's sha {att_sha} for attempt {aid or '?'} (job {job or '?'}) disagree: "
                      "which tree the proxy runs is not established")
    sha = row_sha or att_sha
    if not sha:
        return None, (f"neither the submissions row (extra.sha) nor the attempts ledger (an `attempt` event for attempt {aid or '?'}, job {job or '?'}) records "
                      "the sha the proxy was submitted at")
    return sha, ("the submissions row's extra.sha" + (" (= the attempts ledger's)" if att_sha else "")) if row_sha else f"the attempts ledger's attempt {aid or job}"


def discover_petri_specs(state_root: Optional[str], model: Optional[str], clusters: Mapping[str, Mapping[str, str]],
                         log_state_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """The drive specs as the MODEL STATE stands NOW: every `<state root>/WAVE.<view>.json` (a launched wave; `.partial.json` is
    not one) whose `petri.proxy` unit has a `submitted` row in `<state root>/submissions.jsonl` (the LAST row wins: a re-run binds a new
    job) becomes one spec — tag `<model>[_<view>]`, the proxy's cluster, that cluster's ssh alias and wave root (`--petri-cluster`),
    `<wave root>/petri/<tag>/serving/endpoint.json`, THE UNIT'S SHA (:func:`petri_unit_sha`: the sha that unit's bound submission was made at,
    read off its own submitted row / attempt — never the wave file's, which a re-mint or the overflow pass leaves behind; a unit whose sha
    cannot be established is refused by name, `petri_drive_sha_unresolved`, and yields no spec), and the ledger's job id + name (a cross-check
    against the id the endpoint names). The overflow pass's spec (unit petri.overflow) is resolved the same way, from ITS row. Cheap (three
    small files per wave) and re-read every pass."""
    if not state_root or not model or not os.path.isdir(state_root):
        return []
    out: List[Dict[str, str]] = []
    rows = _read_jsonl(os.path.join(state_root, PETRI_SUBMISSIONS_LEDGER))
    attempts = _read_jsonl(os.path.join(state_root, PETRI_ATTEMPTS_LEDGER))

    def unit_sha(row: Mapping[str, object], utag: str, wave_sha: str) -> Optional[str]:
        """The unit's sha or None (refused by name, said once per (tag, job)); a wave file at another sha is said once per pair, informational."""
        usha, where = petri_unit_sha(row, attempts)
        job = str(row.get("job_id") or "")
        if not usha:
            key = ("shaunresolved", utag, job, where)
            if key not in _PETRI_DISCOVERY_SAID:
                _PETRI_DISCOVERY_SAID.add(key)
                log_event(log_state_dir, {"event": "petri_drive_sha_unresolved", "tag": utag, "job_id": job, "attempt_id": str(row.get("attempt_id") or ""), "wave_sha": wave_sha,
                                          "reason": where, "note": "the drive checks out the sha the unit's bound submission was made at, never WAVE.<view>.json's; "
                                                                   "no spec for this unit until its submissions row (extra.sha) or attempts-ledger attempt names it"})
            return None
        if usha != wave_sha:
            key = ("shamismatch", utag, wave_sha, usha)
            if key not in _PETRI_DISCOVERY_SAID:
                _PETRI_DISCOVERY_SAID.add(key)
                log_event(log_state_dir, {"event": "petri_drive_sha_mismatch", "tag": utag, "wave_sha": wave_sha, "ledger_sha": usha, "job_id": job, "ledger_sha_from": where,
                                          "note": "WAVE.<view>.json's wave_sha differs from the unit's bound submission's sha (a re-minted proxy, or the overflow pass, "
                                                  "which never rewrites the wave file): the drive follows the unit's sha, not the wave file"})
        return usha

    for name in sorted(os.listdir(state_root)):
        if not (name.startswith("WAVE.") and name.endswith(".json")) or name.endswith(".partial.json"):
            continue
        try:
            with open(os.path.join(state_root, name), encoding="utf-8") as fh:
                wave = json.load(fh)
        except (OSError, ValueError):
            continue
        view, sha = str(wave.get("view") or ""), str(wave.get("wave_sha") or "")
        if not view or not sha:
            continue
        tag = model if view == "main" else f"{model}_{view}"
        petri = [r for r in rows if r.get("event") == "submitted" and r.get("unit") == "petri.proxy"
                 and str(r.get("logical_work_id") or "").startswith(f"{model}/{view}/")]
        if not petri:
            key = ("nopetri", tag)
            if key not in _PETRI_DISCOVERY_SAID:
                _PETRI_DISCOVERY_SAID.add(key)
                log_event(log_state_dir, {"event": "petri_drive_no_unit", "tag": tag, "note": "the wave has no submitted petri.proxy unit yet"})
            continue
        row = petri[-1]; label = str(row.get("cluster") or "")
        cl = clusters.get(label)
        if not cl:
            key = ("nocluster", tag, label)
            if key not in _PETRI_DISCOVERY_SAID:
                _PETRI_DISCOVERY_SAID.add(key)
                log_event(log_state_dir, {"event": "petri_drive_no_cluster", "tag": tag, "cluster": label, "note": "no --petri-cluster for the proxy's cluster"})
            continue
        first_sha = unit_sha(row, tag, sha)
        if first_sha:
            out.append({"tag": tag, "model": model, "view": view, "cluster": label, "ssh": cl["ssh"], "sha": first_sha, "job_id": str(row.get("job_id") or ""),
                        "job_name": str(row.get("job_name") or ""), "endpoint": f"{cl['wave_root']}/petri/{tag}/serving/endpoint.json", "shared": cl.get("shared") or ""})
        # the OVERFLOW pass (petri-completion): the LAST `submitted` row of unit petri.overflow for this wave is one more spec — its own tag,
        # record and lock (`<tag>.overflow`), the rerun proxy's endpoint (serving_rerun/), ITS OWN submitted sha (06 --pass overflow submits the
        # re-run proxy at the tree it runs from and never rewrites WAVE), the first pass's bound job as its start gate (resolved on its own: a
        # first pass refused above still gates an overflow whose first-pass record stands)
        over = [r for r in rows if r.get("event") == "submitted" and r.get("unit") == PETRI_OVERFLOW_UNIT
                and str(r.get("logical_work_id") or "").startswith(f"{model}/{view}/")]
        if over:
            otag = f"{tag}{PETRI_OVERFLOW_RECORD_SUFFIX}"
            orow = over[-1]; olabel = str(orow.get("cluster") or ""); ocl = clusters.get(olabel)
            if not ocl:
                key = ("nocluster", otag, olabel)
                if key not in _PETRI_DISCOVERY_SAID:
                    _PETRI_DISCOVERY_SAID.add(key)
                    log_event(log_state_dir, {"event": "petri_drive_no_cluster", "tag": otag, "cluster": olabel, "note": "no --petri-cluster for the overflow proxy's cluster"})
                continue
            over_sha = unit_sha(orow, otag, sha)
            if over_sha:
                out.append({"tag": otag, "base_tag": tag, "pass": PETRI_PASS_OVERFLOW, "model": model, "view": view, "cluster": olabel,
                            "ssh": ocl["ssh"], "sha": over_sha, "job_id": str(orow.get("job_id") or ""), "job_name": str(orow.get("job_name") or ""),
                            "endpoint": f"{ocl['wave_root']}/petri/{tag}/serving_rerun/endpoint.json", "shared": ocl.get("shared") or "",
                            "first_job_id": str(row.get("job_id") or "")})
    return out


def parse_petri_specs(specs: Optional[Sequence[str]]) -> List[Dict[str, str]]:
    """`--petri-drive "tag=<K>_<view>;model=<K>;view=<view>;cluster=<label>;ssh=<alias>;endpoint=<remote path>;sha=<wave sha>"`
    (repeatable; an explicit spec beside the discovered ones); every key required, refused by name otherwise."""
    out: List[Dict[str, str]] = []
    for spec in specs or ():
        d: Dict[str, str] = {}
        for tok in str(spec).split(";"):
            if not tok.strip():
                continue
            if "=" not in tok:
                raise ValueError(f"--petri-drive {spec!r}: token {tok!r} is not key=value")
            k, v = tok.split("=", 1); d[k.strip()] = v.strip()
        missing = [k for k in PETRI_SPEC_KEYS if not d.get(k)]
        if missing:
            raise ValueError(f"--petri-drive {spec!r}: missing {missing} (need {list(PETRI_SPEC_KEYS)})")
        if d.get("pass", PETRI_PASS_FIRST) not in (PETRI_PASS_FIRST, PETRI_PASS_OVERFLOW):
            raise ValueError(f"--petri-drive {spec!r}: pass must be {PETRI_PASS_FIRST} or {PETRI_PASS_OVERFLOW}")
        if d.get("pass") == PETRI_PASS_OVERFLOW and not d.get("first_job_id"):
            raise ValueError(f"--petri-drive {spec!r}: an overflow spec needs first_job_id (the first pass's bound job, its start gate)")
        out.append(d)
    return out


def petri_drive_record_path(state_dir: str, tag: str) -> str:
    return os.path.join(state_dir, f"petri_drive.{tag}.json")


# --- the checkout and the preparation ------------------------------------------------------------------------------------
def petri_checkout(repo: str, sha: str, state_dir: str, python: Optional[str] = None, run=subprocess.run) -> Tuple[Optional[str], Optional[str]]:
    """A PERSISTENT detached checkout of the wave sha under the daemon's state (`<state-dir>/checkouts/<sha>`): made once with
    `git worktree add --detach` from --repo's repository (after a `worktree prune`, so a registered path whose directory is gone does not
    block it; a sha this clone has not fetched is fetched from origin first), verified at HEAD, and reused across passes and restarts.
    `<checkout>/.venv` becomes a directory of links into the venv that owns `python` when the checkout has none (the drive resolves its interpreter as
    `$WT/.venv/bin/python`; `.venv/` is gitignored and outside configs/scripts/src, so the drive's dirty-tree check is untouched). Returns
    (path, None) or (None, why)."""
    path = os.path.join(state_dir, "checkouts", sha)

    def head() -> Optional[str]:
        r = run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None

    def make() -> Optional[str]:
        """Create the checkout; a reason when it cannot. Under the checkouts lock: a second daemon preparing the same sha waits for this one
        and then finds the finished checkout instead of racing `worktree add` (or reading a half-populated tree)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _checkouts_lock(state_dir):
            if os.path.isdir(path):
                return None
            if run(["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, text=True, timeout=60).returncode != 0:
                f = run(["git", "-C", repo, "fetch", "--quiet", "origin"], capture_output=True, text=True, timeout=900)
                if run(["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, text=True, timeout=60).returncode != 0:
                    return (f"wave sha {sha[:12]} is not in {repo}'s repository"
                            + (f" and `git fetch origin` failed: {(f.stderr or f.stdout).strip()[:200]}" if f.returncode != 0
                               else " (not on origin either, or this clone does not carry the development history that commit belongs to)"))
            run(["git", "-C", repo, "worktree", "prune"], capture_output=True, text=True, timeout=120)
            r = run(["git", "-C", repo, "worktree", "add", "--detach", path, sha], capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                return f"git worktree add --detach {path} {sha[:12]} failed: {(r.stderr or r.stdout).strip()[:300]}"
            return None

    try:
        if not os.path.isdir(path):
            why = make()
            if why:
                return None, why
        h = head()
        if h != sha:
            return None, f"checkout {path} is at {(h or '?')[:12]}, not the wave sha {sha[:12]} (remove it and the daemon re-makes it)"
        if python:
            venv_root = os.path.dirname(os.path.dirname(os.path.abspath(python)))
            link = os.path.join(path, ".venv")
            if os.path.isfile(os.path.join(venv_root, "pyvenv.cfg")) and not os.path.lexists(link):
                # a DIRECTORY of links, not a link to the directory: `.gitignore`'s `.venv/` matches directories only, and a linked
                # `.venv` would show as untracked in the checkout's porcelain
                os.makedirs(link)
                for name in os.listdir(venv_root):
                    os.symlink(os.path.join(venv_root, name), os.path.join(link, name))
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"checkout {path}: {type(exc).__name__}: {exc}"
    return path, None


@contextlib.contextmanager
def _claims_lock(state_dir: str):
    """The global claims lock (`<state-dir>/petri_claims.lock`, blocking): every pass holds it around a tag's WHOLE body (probe → claim →
    spawn → started) and the pruner around its whole read-and-remove, so no claim can land between the pruner's reading of the live
    attempts and its `worktree remove` (re-check 28: a second daemon starting a replacement endpoint at the same wave sha while the
    first pruned it deleted the running drive's checkout). Lock order everywhere: claims → tag → checkouts."""
    os.makedirs(state_dir, exist_ok=True)
    fd = os.open(os.path.join(state_dir, "petri_claims.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def _checkouts_lock(state_dir: str):
    """The (blocking) lock over creating and removing checkouts under `<state-dir>/checkouts/`."""
    os.makedirs(os.path.join(state_dir, "checkouts"), exist_ok=True)
    fd = os.open(os.path.join(state_dir, "checkouts", ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _petri_completed_at_sha(rec: Mapping[str, object], sha: str) -> bool:
    """A drive completed for this tag at this wave sha (any endpoint identity): the wave's Petri work is done there."""
    return any(a.get("state") == "completed" and str((a.get("identity") or {}).get("wave_sha")) == sha for a in rec.get("attempts") or [])


def prune_petri_checkouts(*, state_dir: str, repo: str, specs: Sequence[Mapping[str, str]], run=subprocess.run) -> List[str]:
    """Remove every `<state-dir>/checkouts/<sha>` whose wave is CLOSED to the drive: no current spec (a WAVE.<view>.json with its petri unit)
    names the sha any more — the wave was re-frozen at a newer sha, or is gone — or every spec still at that sha has its drive `completed`
    for its tag (any endpoint identity); and no attempt is live (`claimed` / `started`) from that checkout — read under the global claims
    lock, which every claim also holds, so the reading is current at the removal. `git worktree remove --force` (the venv links are
    untracked) + `prune`, one `petri_checkout_pruned` event each. `petri_checkout` re-makes a pruned checkout on demand when a replaced
    proxy needs it, so pruning is always safe; nothing is pruned when no spec is known at all (a transient discovery failure must not
    sweep every checkout). Returns the pruned paths."""
    root = os.path.join(state_dir, "checkouts")
    if not specs or not os.path.isdir(root):
        return []
    with _claims_lock(state_dir):     # no claim can land between this reading of the live attempts and the removal below (re-check 28)
        return _prune_petri_checkouts_locked(root, state_dir=state_dir, repo=repo, specs=specs, run=run)


def _prune_petri_checkouts_locked(root: str, *, state_dir: str, repo: str, specs: Sequence[Mapping[str, str]], run) -> List[str]:
    live: set = set()
    completed_by_sha: Dict[str, set] = {}
    for name in os.listdir(state_dir):
        if not (name.startswith("petri_drive.") and name.endswith(".json")):
            continue
        rec = _petri_load(os.path.join(state_dir, name)) or {}
        for a in rec.get("attempts") or []:
            if a.get("state") in ("claimed", "started") and a.get("checkout"):
                live.add(str(a["checkout"]))
            if a.get("state") == "completed":
                completed_by_sha.setdefault(str((a.get("identity") or {}).get("wave_sha")), set()).add(str(rec.get("tag")))
    tags_by_sha: Dict[str, set] = {}
    for sp in specs:
        tags_by_sha.setdefault(sp["sha"], set()).add(sp["tag"])
    pruned: List[str] = []
    for sha in sorted(os.listdir(root)):
        path = os.path.join(root, sha)
        if not os.path.isdir(path) or path in live:
            continue
        tags = tags_by_sha.get(sha)
        if tags is None:
            why = "no current wave names this sha"
        elif tags <= completed_by_sha.get(sha, set()):
            why = f"the drive completed for every wave at this sha ({', '.join(sorted(tags))})"
        else:
            continue
        try:
            with _checkouts_lock(state_dir):
                r = run(["git", "-C", repo, "worktree", "remove", "--force", path], capture_output=True, text=True, timeout=600)
                if r.returncode != 0 and os.path.isdir(path):
                    _say_once(state_dir, ("prune_failed", path), {"event": "petri_checkout_prune_failed", "checkout": path, "wave_sha": sha,
                                                                  "error": (r.stderr or r.stdout).strip()[:300]})
                    continue
                run(["git", "-C", repo, "worktree", "prune"], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            _say_once(state_dir, ("prune_failed", path), {"event": "petri_checkout_prune_failed", "checkout": path, "wave_sha": sha, "error": f"{type(exc).__name__}: {exc}"})
            continue
        pruned.append(path)
        log_event(state_dir, {"event": "petri_checkout_pruned", "checkout": path, "wave_sha": sha, "reason": why})
    return pruned


_PETRI_PY_FIT: Dict[str, Optional[str]] = {}
_PETRI_PREPARED: set = set()


def petri_prepare(spec: Mapping[str, str], *, checkout: str, python: str, bank_root: str, bundle_dir: str, roster: str,
                  run=subprocess.run) -> Tuple[Optional[str], str]:
    """Everything the drive refuses without, checked or BUILT before a start (a refusal here is a reason, never a dead attempt): the
    interpreter imports inspect_ai + petri_bloom (once per interpreter); the frozen scenario bundle is present, or installed from the copy
    the checkout publishes (petri_install_bundle; note "scenario bundle installed from the checkout"); the grading host bank view of
    (model, view) — `<bank root>/<model>/cells_view/<view>` with its VIEW_MANIFEST.json — verifies under the checkout's make_bank_view.py
    --check, else it is built there and verified (a prompted-only view stages nothing and still needs its manifest directory). Returns
    (why-not, note): (None, "") when ready, (None, "bank view built") after a build, (reason, "") otherwise. Success is cached per
    (checkout, model, view, interpreter) for the life of the process."""
    key = (checkout, spec["model"], spec["view"], python, bank_root, bundle_dir, roster)
    if key in _PETRI_PREPARED:
        return None, ""
    if python not in _PETRI_PY_FIT:
        try:
            r = run([python, "-c", PETRI_IMPORT_CHECK], capture_output=True, text=True, timeout=300)
            _PETRI_PY_FIT[python] = None if r.returncode == 0 else f"{python} lacks what the drive imports ({PETRI_IMPORT_CHECK}): {(r.stderr or r.stdout).strip()[:200]}"
        except (OSError, subprocess.SubprocessError) as exc:
            _PETRI_PY_FIT[python] = f"{python} could not run: {exc}"
    if _PETRI_PY_FIT[python]:
        return _PETRI_PY_FIT[python], ""
    notes = []
    if not os.path.isdir(bundle_dir):
        why = petri_install_bundle(checkout=checkout, python=python, bundle_dir=bundle_dir, run=run)
        if why:
            return why, ""
        notes.append("scenario bundle installed from the checkout")
    if "$" in bank_root:
        return (f"the Petri bank root {bank_root} names a site directory this host does not resolve: set SISTEER_STAGE_DIR in the "
                "site config, or pass --petri-bank-root"), ""
    bank = os.path.join(bank_root, spec["model"]); view_dir = os.path.join(bank, "cells_view", spec["view"])
    manifest = os.path.join(view_dir, "VIEW_MANIFEST.json")
    mbv = os.path.join(checkout, "scripts", "sdfbase", "full22", "make_bank_view.py")
    if not os.path.isfile(mbv):
        return f"no make_bank_view.py at {mbv}", ""
    roster_path = roster if os.path.isabs(roster) else os.path.join(checkout, roster)
    base = [python, mbv, "--bank", bank, "--target", spec["model"], "--view", spec["view"], "--roster", roster_path]
    env = dict(os.environ, PYTHONPATH=os.path.join(checkout, "src"))
    try:
        ok = os.path.isfile(manifest) and run(base + ["--check"], cwd=checkout, env=env, capture_output=True, text=True, timeout=900).returncode == 0
        if not ok:
            os.makedirs(bank, exist_ok=True)
            b = run(base, cwd=checkout, env=env, capture_output=True, text=True, timeout=1800)
            if b.returncode != 0:
                return f"the grading host bank view {view_dir} could not be built: {(b.stdout or b.stderr).strip()[:300]}", ""
            c = run(base + ["--check"], cwd=checkout, env=env, capture_output=True, text=True, timeout=900)
            if c.returncode != 0 or not os.path.isfile(manifest):
                return f"the grading host bank view {view_dir} does not verify after its build: {(c.stdout or c.stderr).strip()[:300]}", ""
            notes.append("bank view built")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"make_bank_view.py could not run: {exc}", ""
    _PETRI_PREPARED.add(key)
    return None, "; ".join(notes)


def petri_install_bundle(*, checkout: str, python: str, bundle_dir: str, run=subprocess.run) -> Optional[str]:
    """A grading host with no scenario store: the wave checkout publishes the frozen bundle at `data/petri/scenarios/<hash>` (shipped for
    exact reproduction), and the checkout's own loader installs it into `bundle_dir` (`python -m sisteer.capeval.petri.scenarios install`:
    the copy verified against the checkout's lock file by file, laid into a mode-0700 read-only store and verified again; an existing store is
    never rewritten). The store is content-addressed, so the copy installed is the one named like `bundle_dir`. None when installed, else the
    reason (the drive refuses without the store)."""
    h = os.path.basename(os.path.normpath(bundle_dir))
    src = os.path.join(checkout, "data", "petri", "scenarios", h)
    if not os.path.isdir(src):
        return f"no Petri scenario bundle at {bundle_dir}, and the checkout publishes none at {src}"
    cmd = [python, "-m", "sisteer.capeval.petri.scenarios", "install", "--dest", bundle_dir, "--from", src,
           "--lock", os.path.join(checkout, "configs", "capeval", "petri-scenarios.lock.yaml")]
    try:
        r = run(cmd, cwd=checkout, env=dict(os.environ, PYTHONPATH=os.path.join(checkout, "src")), capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"no Petri scenario bundle at {bundle_dir}; installing the checkout's copy {src} could not run: {exc}"
    if r.returncode != 0 or not os.path.isdir(bundle_dir):
        return f"no Petri scenario bundle at {bundle_dir}; the checkout's copy {src} did not install: {(r.stderr or r.stdout).strip()[:300]}"
    return None


_PETRI_SAID: set = set()


def _say_once(state_dir: Optional[str], key: tuple, event: Mapping[str, object]) -> None:
    """One event per distinct key for the life of the process: the pass log stays readable across the minutes a proxy takes to stand up."""
    if key in _PETRI_SAID:
        return
    _PETRI_SAID.add(key)
    log_event(state_dir, event)


def _petri_prepare_spec(spec: Mapping[str, str], *, state_dir: str, repo: str, python: str, bank_root: str, bundle_dir: str, roster: str,
                        run=subprocess.run) -> Optional[str]:
    """The checkout at the wave sha, then the preparation; the checkout path when ready, None (with a `petri_drive_unprepared` event, once
    per reason PER WAVE SHA) otherwise. Run eagerly while the endpoint is absent so the start, when the proxy is up, is immediate. The
    once-keys carry the wave sha (demo-import, 2026-10-11): a wave re-launched at a new sha (06 --resubmit rewrites WAVE.<view>.json) that
    fails to prepare for the SAME reason is a new fact and logs again; before this the key was (tag, reason) alone and the daemon's log
    stayed silent through the re-launch, which read as a stuck discovery."""
    tag = spec["tag"]
    checkout, why = petri_checkout(repo, spec["sha"], state_dir, python=python, run=run)
    if checkout is None:
        _say_once(state_dir, ("unprepared", tag, spec["sha"], why), {"event": "petri_drive_unprepared", "tag": tag, "wave_sha": spec["sha"], "reason": why})
        return None
    why, note = petri_prepare(spec, checkout=checkout, python=python, bank_root=bank_root, bundle_dir=bundle_dir, roster=roster, run=run)
    if why:
        _say_once(state_dir, ("unprepared", tag, spec["sha"], why), {"event": "petri_drive_unprepared", "tag": tag, "wave_sha": spec["sha"], "checkout": checkout, "reason": why})
        return None
    _say_once(state_dir, ("prepared", tag, spec["sha"], checkout), {"event": "petri_drive_prepared", "tag": tag, "wave_sha": spec["sha"], "checkout": checkout, "python": python,
                                                       "bank_view": os.path.join(bank_root, spec["model"], "cells_view", spec["view"]), "note": note or "verified"})
    return checkout


# --- the endpoint, the proxy job, the identity -----------------------------------------------------------------------------
def _ssh_argv(ssh_bin: str, alias: str, remote: str) -> List[str]:
    return [ssh_bin, "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", alias, remote]


def _petri_probe_endpoint(run, ssh_bin: str, alias: str, endpoint: str) -> Tuple[str, Optional[str], Optional[dict], str]:
    """One ssh: does the endpoint file exist, and if so its mtime (seconds) and content. ('absent' | 'present' | 'unreachable' | 'error',
    mtime, content, detail)."""
    q = shlex.quote(endpoint)
    try:
        r = run(_ssh_argv(ssh_bin, alias, f"test -f {q} || exit 3; stat -c %Y {q}; cat {q}"), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        return "unreachable", None, None, f"{type(exc).__name__}: {exc}"
    if r.returncode == 3:
        return "absent", None, None, ""
    if r.returncode == 255:
        return "unreachable", None, None, f"ssh {alias} exited 255"
    if r.returncode != 0:
        return "error", None, None, f"probe exited {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
    lines = r.stdout.splitlines()
    mtime = lines[0].strip() if lines else ""
    try:
        content = json.loads("\n".join(lines[1:]) or "{}")
    except ValueError:
        content = None
    if not mtime.isdigit() or not isinstance(content, dict):
        return "error", mtime or None, None, "endpoint.json unreadable (a torn write? read again next pass)"
    return "present", mtime, content, ""


def _petri_job_state(run, ssh_bin: str, alias: str, job_id: str) -> Tuple[str, str]:
    """The proxy job's Slurm state through `squeue -j <id> -h -o %T` over the same ssh: 'RUNNING' (alive — the only state that serves),
    another state, 'gone' (not in the queue: terminal), 'unreachable' (ssh), or 'unknown' (squeue itself failed). The drive holds the job
    to RUNNING by the same command."""
    try:
        r = run(_ssh_argv(ssh_bin, alias, f"squeue -j {shlex.quote(job_id)} -h -o %T"), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        return "unreachable", f"{type(exc).__name__}: {exc}"
    if r.returncode == 255:
        return "unreachable", f"ssh {alias} exited 255"
    states = r.stdout.split()
    if r.returncode == 0:
        return ("RUNNING" if "RUNNING" in states else (states[0] if states else "gone")), ""
    if "invalid job id" in (r.stderr or "").lower():
        return "gone", (r.stderr or "").strip()[:200]
    return "unknown", f"squeue exited {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"


def _petri_fetch_expected(run, ssh_bin: str, alias: str, shared_root: str, job_id: str, cells_served: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    """The wave's EXPECTED run set for the petri unit under this proxy job, independent of any drive (re-check 4): the proxy's job-start
    registration `<shared root>/registry/pending/<job id>.json` (f22_registry_register: one run per (eval, object)), and the objects the
    standup skipped by name (`cells_served.json.cells_skipped`, at the path endpoint.json names) — runs of those objects can never be filed
    under this endpoint and are excluded from what completion requires. Returns (the endpoint entry, None) or (None, why)."""
    if not shared_root:
        return None, "no shared root for the proxy's cluster (give --petri-cluster its 4th part): cannot read the job's registration"
    pend = f"{shared_root}/registry/pending/{job_id}.json"
    try:
        r = run(_ssh_argv(ssh_bin, alias, f"cat {shlex.quote(pend)}"), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"reading {alias}:{pend}: {type(exc).__name__}: {exc}"
    if r.returncode != 0:
        return None, f"no job-start registration at {alias}:{pend} (rc {r.returncode}): the proxy registered no runs — nothing a drive files could complete"
    try:
        doc = json.loads(r.stdout)
        runs = doc.get("runs") if isinstance(doc, dict) else None
        expected = [{"run_id": str(x["run_id"]), "eval_id": str((x.get("route") or {}).get("eval_id") or (x.get("plan") or {}).get("eval_id") or ""),
                     "object": str((x.get("route") or {}).get("object") or "")} for x in runs or []]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return None, f"{alias}:{pend} is not a registration document: {type(exc).__name__}: {exc}"
    if not expected:
        return None, f"{alias}:{pend} names no runs"
    skipped_objects: List[str] = []
    if cells_served:
        try:
            r2 = run(_ssh_argv(ssh_bin, alias, f"cat {shlex.quote(str(cells_served))}"), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
            if r2.returncode == 0:
                cs = json.loads(r2.stdout)
                skipped_objects = [str(c.get("cell_id")) for c in (cs.get("cells_skipped") or []) if isinstance(c, dict) and c.get("cell_id")]
        except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
            skipped_objects = []
    skipped_runs = sorted({e["run_id"] for e in expected if e["object"] in set(skipped_objects)})
    return {"registration": pend, "expected": expected, "expected_ids": sorted(e["run_id"] for e in expected), "skipped_objects": skipped_objects,
            "skipped_runs": skipped_runs, "covered": [], "complete": False, "fetched_utc": utc_now()}, None


def _petri_identity(job_id: str, mtime: str, sha: str) -> Dict[str, str]:
    return {"job_id": str(job_id), "endpoint_mtime": str(mtime), "wave_sha": sha}


def _identity_key(ident: Optional[Mapping[str, object]]) -> str:
    ident = ident or {}
    return f"{ident.get('job_id')}@{ident.get('endpoint_mtime')}@{ident.get('wave_sha')}"


# --- the record, the supervision, the claim (findings 11 + 13) -------------------------------------------------------------------------
_PETRI_PROCS: Dict[str, object] = {}       # tag -> the Popen this process spawned (its exit code by poll); a restarted daemon uses /proc + the rc file


def _petri_load(rec_path: str) -> Optional[dict]:
    try:
        with open(rec_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_json(path: Optional[str]) -> Optional[dict]:
    doc = _petri_load(path) if path else None
    return doc if isinstance(doc, dict) else None


def _petri_save(rec_path: str, rec: Mapping[str, object]) -> None:
    tmp = rec_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True); fh.write("\n")
    os.replace(tmp, rec_path)


def _petri_new_record(spec: Mapping[str, str]) -> dict:
    return {"tag": spec["tag"], "model": spec["model"], "view": spec["view"], "cluster": spec["cluster"], "wave_sha": spec["sha"],
            "endpoint": spec["endpoint"], "ledger_job_id": spec.get("job_id") or None, "cap": PETRI_DRIVE_ATTEMPT_CAP, "attempts": []}


def _proc_stat(pid: int) -> Tuple[Optional[str], Optional[int]]:
    """(state, start time in clock ticks) from /proc/<pid>/stat (fields 3 and 22): the start time with the pid identifies the process
    across a pid reuse; the state tells a zombie (exited, not yet reaped) from a live process."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            rest = fh.read().rsplit(")", 1)[1].split()
        return rest[0], int(rest[19])
    except (OSError, ValueError, IndexError):
        return None, None


def _proc_start(pid: int) -> Optional[int]:
    return _proc_stat(pid)[1]


def _proc_alive(pid: int, start: Optional[int]) -> bool:
    state, s = _proc_stat(pid)
    return s is not None and state != "Z" and (start is None or s == start)


def _read_rc(rc_file: Optional[str]) -> Optional[int]:
    try:
        with open(rc_file or "", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError, TypeError):
        return None


def _read_marker(pid_file: Optional[str]) -> Optional[dict]:
    """The wrapper's pid marker ({pid, proc_start, utc}) when it exists and names a pid."""
    m = _read_json(pid_file)
    return m if m is not None and isinstance(m.get("pid"), int) else None


def _wait_marker(pid_file: str, proc, timeout: float = PETRI_MARKER_WAIT_S) -> Optional[dict]:
    """The marker is the wrapper's first action: wait for it (or for the child to have exited, which settles the matter either way)."""
    end = time.time() + timeout
    while True:
        m = _read_marker(pid_file)
        if m is not None:
            return m
        if proc.poll() is not None:
            return _read_marker(pid_file)
        if time.time() >= end:
            return None
        time.sleep(0.02)


def _spec_pass(spec: Mapping[str, object]) -> str:
    return str(spec.get("pass") or PETRI_PASS_FIRST)


def petri_overflow_gate(state_dir: Optional[str], spec: Mapping[str, str]) -> Tuple[bool, str]:
    """The overflow pass's START GATE (petri-completion): (True, why) when the FIRST pass's record (`petri_drive.<base tag>.json`) holds an
    attempt for `first_job_id` in state `incomplete` and none `claimed|started`; (False, why) otherwise — the first pass has not settled
    incomplete (nothing is ledgered yet, or its drive is live), so the overflow drive would find nothing to re-run or race it."""
    base = str(spec.get("base_tag") or str(spec["tag"]).rsplit(PETRI_OVERFLOW_RECORD_SUFFIX, 1)[0])
    first_job = str(spec.get("first_job_id") or "")
    rec = _petri_load(petri_drive_record_path(str(state_dir), base)) if state_dir else None
    if not isinstance(rec, dict):
        return False, f"no first-pass record petri_drive.{base}.json yet"
    same = [a for a in (rec.get("attempts") or []) if str((a.get("identity") or {}).get("job_id")) == first_job]
    if not same:
        return False, f"the first-pass record holds no attempt for its bound job {first_job}"
    live = [a for a in same if a.get("state") in ("claimed", "started")]
    if live:
        return False, f"a first-pass drive of job {first_job} is live (attempt {live[-1].get('n')} {live[-1].get('state')})"
    if not any(a.get("state") == "incomplete" for a in same):
        return False, f"no first-pass attempt of job {first_job} settled incomplete (states {sorted({str(a.get('state')) for a in same})})"
    return True, f"first-pass job {first_job} settled incomplete"


def _petri_verify_overflow(rec: dict, att: Mapping[str, object], doc: dict) -> Tuple[str, Optional[str], Optional[dict]]:
    """An OVERFLOW attempt's exit 0: the drive's verdict complete, rc 0, hand-off on (checked by the caller); then —
    `handoff.verify_scope == control_only` (every selected leg a control leg: no registered run in scope, verify not run): `runs_planned`
    0 is ACCEPTED iff the result's `legs.detail` shows every in-scope leg with `control_of` set and `status == complete` — a control leg is
    not a registered object, so its completion is the drive's verdict, read off the record the drive wrote, never the registry; required = ∅;
    otherwise (`selected` / `all`): planned > 0, filed == planned, verify_rc 0 as today, and required = expected − skipped − OUT-OF-SCOPE
    (the summary's `scope: out` rows: the registered runs of cells the proxy declared but the drive did not select — `none`'s, declared so
    the controls twin is attested and never driven, and on a resubmit the cells an earlier overflow drive already completed, credited by
    their registry state). A first-pass attempt never takes this path."""
    ho = doc.get("handoff") or {}
    scope = ho.get("verify_scope")
    ep = (rec.get("endpoints") or {}).get(_identity_key(att.get("identity")))
    if not ep:
        return "partial", "partial:?/? — the drive filed its scope but this endpoint's expected run set is not recorded (claimed before this contract): never `completed` without it (an error, counted toward the cap)", doc
    planned, filed = int(ho.get("runs_planned") or 0), int(ho.get("runs_filed") or 0)
    if scope == PETRI_VERIFY_SCOPE_CONTROL_ONLY:
        detail = [d for d in ((doc.get("legs") or {}).get("detail") or []) if isinstance(d, dict) and d.get("in_scope")]
        if planned != 0 or ho.get("summary"):
            return "failed", f"exit 0 with verify_scope control_only but runs_planned {planned} / a verify summary {ho.get('summary')!r}: a control-only pass verifies nothing", doc
        bad = [d for d in detail if not d.get("control_of") or d.get("status") != "complete"]
        if not detail or bad:
            return "failed", ("exit 0 with no registered run in scope but " + (f"a non-control / incomplete leg in scope ({[(d.get('idx'), d.get('cell_id'), d.get('status')) for d in bad]})" if bad else "no in-scope leg in legs.detail")), doc
        ep["covered"] = sorted(set(ep.get("covered") or [])); ep["out_of_scope"] = sorted(set(ep.get("expected_ids") or []))
        att["coverage"] = {"filed_by_this_drive": 0, "covered": 0, "required": 0, "expected": len(ep.get("expected_ids") or []), "skipped": len(ep.get("skipped_runs") or []),
                           "out_of_scope": len(ep["out_of_scope"]), "remaining": 0, "verify_scope": scope, "control_legs": [d.get("cell_id") for d in detail]}  # type: ignore[index]
        return "completed", None, doc
    if scope not in ("selected", "all"):
        return "failed", f"exit 0 but the summary names no verify scope ({scope!r}): nothing says which runs were filed", doc
    if planned <= 0 or filed != planned or str(ho.get("verify_rc")) != "0":
        return "failed", f"exit 0 but only {filed} of {planned} planned run(s) verified filed on this host's spool (verify_rc={ho.get('verify_rc')!r})", doc
    vs = _read_json(ho.get("summary"))
    rows = vs.get("runs") if vs else None
    if not isinstance(rows, list):
        return "failed", f"exit 0 but no verify summary at {ho.get('summary')!r}: which runs were filed is unknown", doc
    filed_ids = {str(r.get("run_id")) for r in rows if isinstance(r, dict) and r.get("filed")}
    skipped_ids = {str(r.get("run_id")) for r in rows if isinstance(r, dict) and r.get("scope") == "skipped"}
    out_ids = {str(r.get("run_id")) for r in rows if isinstance(r, dict) and r.get("scope") == "out"}
    ep["covered"] = sorted(set(ep.get("covered") or []) | filed_ids)
    ep["skipped_runs"] = sorted(set(ep.get("skipped_runs") or []) | skipped_ids)
    ep["out_of_scope"] = sorted(out_ids)
    required = set(ep.get("expected_ids") or []) - set(ep["skipped_runs"]) - out_ids
    remaining = sorted(required - set(ep["covered"]))
    ep["remaining"] = remaining; ep["complete"] = not remaining
    cov = {"filed_by_this_drive": len(filed_ids), "covered": len(set(ep["covered"]) & required), "required": len(required), "expected": len(ep.get("expected_ids") or []),
           "skipped": len(ep["skipped_runs"]), "out_of_scope": len(out_ids), "remaining": len(remaining), "verify_scope": scope}
    att["coverage"] = cov  # type: ignore[index]
    if ep["complete"]:
        return "completed", None, doc
    return "partial", (f"partial:{cov['covered']}/{cov['required']} — the overflow drive filed {cov['filed_by_this_drive']} run(s) but this endpoint requires {cov['required']} "
                       f"(expected {cov['expected']} − skipped {cov['skipped']} − out of scope {cov['out_of_scope']}): {cov['remaining']} run(s) uncovered (an ERROR; counted toward the cap)"), doc


def _petri_verify_result(rec: dict, att: Mapping[str, object]) -> Tuple[str, Optional[str], Optional[dict]]:
    """Exit 0 is `completed` only when it is earned: the drive's result summary says verdict `complete`, rc 0, and its
    own scope verified filed (runs_filed == runs_planned > 0, verify_rc 0); then the runs its verify summary shows FILED are added to the
    ENDPOINT's coverage (`rec["endpoints"][identity]`, whose expected set came from the proxy job's registration, never from a drive) and
    the endpoint is complete only when every expected run not skipped by the standup is covered. The daemon's drive is the FULL required
    scope, so a shortfall is an ERROR: `partial:<covered>/<required>` — recorded with its coverage, counted toward the cap like a failure.
    Hand-off `off` / `disabled` is the drive's own recorded exception (coverage unverifiable by design). Returns (state, reason, the summary)."""
    doc = _read_json(att.get("result_file"))  # type: ignore[arg-type]
    if doc is None:
        return "failed", "exit 0 but no result summary: the drive predates this contract (re-freeze the wave at a newer sha) or died before its verdict", None
    ho = doc.get("handoff") or {}
    if doc.get("verdict") != "complete" or str(doc.get("rc")) != "0":
        return "failed", f"exit 0 but the drive's own verdict is {doc.get('verdict')!r} (rc {doc.get('rc')!r})", doc
    if ho.get("mode") in ("off", "disabled"):
        return "completed", f"registry hand-off {ho.get('mode')}: the drive's own exception — coverage unverifiable by design", doc
    if ho.get("mode") != "on":
        return "failed", f"exit 0 but the summary names no hand-off mode ({ho.get('mode')!r}): nothing says the runs were filed", doc
    if att.get("pass") == PETRI_PASS_OVERFLOW:
        return _petri_verify_overflow(rec, att, doc)       # petri-completion: out-of-scope runs and a control-only pass; the first pass below is byte-identical
    planned, filed = int(ho.get("runs_planned") or 0), int(ho.get("runs_filed") or 0)
    if planned <= 0 or filed != planned or str(ho.get("verify_rc")) != "0":
        return "failed", f"exit 0 but only {filed} of {planned} planned run(s) verified filed on this host's spool (verify_rc={ho.get('verify_rc')!r})", doc
    vs = _read_json(ho.get("summary"))
    rows = vs.get("runs") if vs else None
    if not isinstance(rows, list):
        return "failed", f"exit 0 but no verify summary at {ho.get('summary')!r}: which runs were filed is unknown", doc
    filed_ids = {str(r.get("run_id")) for r in rows if isinstance(r, dict) and r.get("filed")}
    skipped_ids = {str(r.get("run_id")) for r in rows if isinstance(r, dict) and r.get("scope") == "skipped"}
    ep = (rec.get("endpoints") or {}).get(_identity_key(att.get("identity")))
    if not ep:
        return "partial", "partial:?/? — the drive filed its scope but this endpoint's expected run set is not recorded (claimed before this contract): never `completed` without it (an error, counted toward the cap)", doc
    ep["covered"] = sorted(set(ep.get("covered") or []) | filed_ids)
    ep["skipped_runs"] = sorted(set(ep.get("skipped_runs") or []) | skipped_ids)
    required = set(ep.get("expected_ids") or []) - set(ep["skipped_runs"])
    remaining = sorted(required - set(ep["covered"]))
    ep["remaining"] = remaining; ep["complete"] = not remaining
    cov = {"filed_by_this_drive": len(filed_ids), "covered": len(set(ep["covered"]) & required), "required": len(required), "expected": len(ep.get("expected_ids") or []),
           "skipped": len(ep["skipped_runs"]), "remaining": len(remaining)}
    att["coverage"] = cov  # type: ignore[index]
    if ep["complete"]:
        return "completed", None, doc
    return "partial", (f"partial:{cov['covered']}/{cov['required']} — the drive filed {cov['filed_by_this_drive']} run(s) but this endpoint requires {cov['required']}: "
                       f"{cov['remaining']} run(s) uncovered (an ERROR: the daemon's drive is the full required scope; counted toward the cap)"), doc


def _petri_reconcile(rec: dict, state_dir: Optional[str]) -> Tuple[Optional[dict], bool]:
    """Settle the record's live attempt — returns (the attempt still running or None, whether the record changed). A `claimed` attempt (or a
    `started` one with no pid) is ADOPTED from the wrapper's pid marker when it exists (the spawning daemon died before its `started`
    save — re-check 2); with no marker it was never spawned: `failed`. A started attempt whose process is gone is settled by its exit
    code — Popen.poll when this process spawned it, else the wrapper's rc file; exit 0 through the result summary (`completed` or
    `failed`, re-check 1), exit 3 `incomplete`, anything else `failed` — and, whatever the code, a federated drive whose DRIVE_START
    attestation was refused settles `unattested` (:func:`_petri_attestation_refusal`; round 1 on 7778694d0), its running drive having been
    stopped at the first pass that read it. Nothing here starts anything. Callers hold the tag's lock."""
    att = rec.get("attempts") or []
    if not att:
        return None, False
    cur = att[-1]
    if cur.get("state") not in ("claimed", "started"):
        return None, False
    tag = rec.get("tag"); changed = False
    if cur["state"] == "claimed" or not isinstance(cur.get("pid"), int):
        marker = _read_marker(cur.get("pid_file") or f"{cur.get('log')}.pid")
        if marker is None:
            cur.update({"state": "failed", "exited_utc": utc_now(), "exit_code": None,
                        "reason": "claimed but never started (no pid marker: a crash between the claim and the spawn)"})
            log_event(state_dir, {"event": "petri_drive_exited", "tag": tag, "attempt": cur.get("n"), "pid": None, "exit_code": None, "state": "failed",
                                  "log": cur.get("log"), "identity": cur.get("identity"), "reason": cur["reason"], "note": "retried on a later pass under the cap"})
            return None, True
        cur.update({"state": "started", "pid": marker["pid"], "proc_start": marker.get("proc_start"), "started_utc": cur.get("started_utc") or marker.get("utc") or utc_now(),
                    "adopted": True})
        changed = True
        log_event(state_dir, {"event": "petri_drive_adopted", "tag": tag, "attempt": cur.get("n"), "pid": marker["pid"], "log": cur.get("log"),
                              "note": "the wrapper's pid marker names a spawned drive the claiming daemon never recorded: adopted, never started twice"})
    pid = cur["pid"]; code: Optional[int] = None
    proc = _PETRI_PROCS.get(str(tag))
    if proc is not None and getattr(proc, "pid", None) == pid:
        code = proc.poll()  # type: ignore[attr-defined]
        if code is None:
            changed = _petri_attest_key(cur, state_dir, tag) or changed
            return cur, _petri_stop_unattested(cur, state_dir, tag) or changed   # a refused attestation stops the running drive at once
        _PETRI_PROCS.pop(str(tag), None)
    else:
        st, start = _proc_stat(pid)
        if st == "Z" and (cur.get("proc_start") is None or start == cur.get("proc_start")):
            try:
                os.waitpid(pid, os.WNOHANG)   # a child inherited across this process's own re-exec (its Popen is gone): reaped, or it stays a zombie for the daemon's life
            except ChildProcessError:
                pass
        elif _proc_alive(pid, cur.get("proc_start")):
            changed = _petri_attest_key(cur, state_dir, tag) or changed
            return cur, _petri_stop_unattested(cur, state_dir, tag) or changed
        code = _read_rc(cur.get("rc_file"))
    # the attestation is read BEFORE the state is decided (an attempt that exited between two passes has not been read yet): under wif a refused
    # one is the verdict -- never through _petri_verify_result, so the endpoint's coverage is never credited with runs filed under an unproven pool
    _petri_attest_key(cur, state_dir, tag)
    refusal = _petri_attestation_refusal(cur, settling=True)
    result: Optional[dict] = None
    if refusal is not None:
        state, reason, result = PETRI_ATTEMPT_UNATTESTED, refusal, _read_json(cur.get("result_file"))
    elif code == 0:
        state, reason, result = _petri_verify_result(rec, cur)
    elif code == PETRI_DRIVE_RC_OVERFLOW:
        state, reason, result = "incomplete", ("legs INCOMPLETE on overflow ledgers and none failed: the proxy is released; the re-run pass "
                                               "(F22_OVERFLOW_RERUN=1) is a deliberate step, not an automatic retry"), _read_json(cur.get("result_file"))
    elif code == PETRI_DRIVE_RC_LOCKS_HELD:
        state, reason, result = "blocked", ("the drive refused by name: a previous attempt's legs still hold the per-leg locks (not counted toward the cap; "
                                            "the survivors of that attempt's session are stopped before the next start)"), _read_json(cur.get("result_file"))
    elif code is None:
        state, reason = "failed", "exited without an rc (killed, or the wrapper never ran)"
    else:
        state, reason, result = "failed", None, _read_json(cur.get("result_file"))
    cur.update({"state": state, "exited_utc": utc_now(), "exit_code": code, "reason": reason,
                "result": {k: result.get(k) for k in ("verdict", "rc", "legs", "handoff", "log")} if result else None})
    n_failed_same = len([a for a in att if a.get("state") in ("failed", "partial") and _identity_key(a.get("identity")) == _identity_key(cur.get("identity"))])
    note = {"completed": "the drive finished and every expected run of this endpoint is filed: never started again for this endpoint",
            "incomplete": "not a success and not retried: drive the overflow re-run pass by hand",
            PETRI_ATTEMPT_UNATTESTED: "the drive's pool is not proven under SISTEER_SECRETS_MODE=wif: not a success, not counted toward the cap and not retried for this "
                                      "endpoint (a retry would attest the same); re-freeze the wave at a sha carrying this mode's attestation or fix the ids file, "
                                      "then edit or delete the record",
            "blocked": "refused over a previous attempt's live legs: not an attempt against the endpoint, not counted; started again once they are gone"}.get(
        state, f"attempt {n_failed_same} of {PETRI_DRIVE_ATTEMPT_CAP} for this endpoint {'fell short' if state == 'partial' else 'failed'}: retried on a later pass while the proxy is alive"
        if n_failed_same < PETRI_DRIVE_ATTEMPT_CAP else f"{PETRI_DRIVE_ATTEMPT_CAP} attempts failed or fell short for this endpoint: no further automatic start")
    log_event(state_dir, {"event": "petri_drive_exited", "tag": tag, "attempt": cur.get("n"), "pid": pid, "exit_code": code, "state": state,
                          "log": cur.get("log"), "result": cur.get("result_file"), "identity": cur.get("identity"), "coverage": cur.get("coverage"), "reason": reason, "note": note})
    _petri_stop_survivors(cur, state_dir, tag)     # the wrapper is gone; its legs may not be (they hold the per-leg locks)
    return None, True


@contextlib.contextmanager
def _petri_lock(state_dir: str, tag: str):
    """The per-tag claim lock (`<state-dir>/petri_drive.<tag>.lock`, flock, non-blocking): yields the lock's fd when held (the spawner
    passes it to the child, which closes it after publishing its pid marker), None when another process holds it — the caller then skips
    this pass, never waits. Released by CLOSING the fd, never by LOCK_UN: an explicit unlock acts on the whole open file description, the
    child's inherited copy included, and would free the lock before the marker exists on any exception path out of the claim window
    (a daemon that dies closes its fd the same way, so the two paths behave alike: the lock outlives the daemon until the child lets go)."""
    fd = os.open(os.path.join(state_dir, f"petri_drive.{tag}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield None
            return
        yield fd
    finally:
        os.close(fd)


def _petri_verdict(rec: Mapping[str, object], key: str) -> Optional[str]:
    """'completed' / 'incomplete' / 'unattested' / 'capped' when the record already settles this endpoint identity, else None (a start is
    allowed): the endpoint is `completed` when its coverage entry is complete (or an attempt was completed under a hand-off that verifies
    nothing — off / disabled); `unattested` when a federated drive's attestation was refused (never retried: a retry at the same wave sha
    attests the same); `failed` and `partial` attempts both count toward the cap (a shortfall of the full-scope drive is an error)."""
    same = [a for a in rec.get("attempts") or [] if _identity_key(a.get("identity")) == key]
    ep = (rec.get("endpoints") or {}).get(key) or {}
    if ep.get("complete") or any(a.get("state") == "completed" for a in same):
        return "completed"
    if any(a.get("state") == PETRI_ATTEMPT_UNATTESTED for a in same):
        return PETRI_ATTEMPT_UNATTESTED
    if any(a.get("state") == "incomplete" for a in same):
        return "incomplete"
    if len([a for a in same if a.get("state") in ("failed", "partial")]) >= PETRI_DRIVE_ATTEMPT_CAP:
        return "capped"
    return None


def _petri_start_held(state_dir: Optional[str], registry_root: Optional[str], *, tag: str, ident: Mapping[str, object], spec: Mapping[str, str],
                      summary: Dict[str, int], where: str) -> bool:
    """The ONE start-hold check of the Petri pass, called at every point a hold must still be absent -- the start decision, right before
    the durable claim (the preparation between them -- the bank view's import / build / verify, the expected-set SSH reads -- takes minutes,
    and a `dispatch.hold` or a schema drift landing meanwhile must stop a startable endpoint) and right before the spawn. True = held:
    counted once per spec (`summary["held"]`; each caller skips the rest), one `petri_drive_held` per tag per hold naming the reason and
    WHERE it was seen (`_say_once`: said again after a hold lifts and returns), nothing claimed, nothing spawned -- the endpoint stays
    startable for the next pass. False = no hold now: the tag's said-keys are dropped so a later hold is said again. The family hold's
    scope (dispatch groups only) stays: it is never read here."""
    hold_kind, hold_why = petri_start_hold(state_dir, registry_root)
    if hold_kind is None:
        for _kind in ("operator", "schema"):
            _PETRI_SAID.discard(("held", tag, _kind))
        return False
    summary["held"] += 1
    _say_once(state_dir, ("held", tag, hold_kind), {"event": "petri_drive_held", "tag": tag, "hold": hold_kind, "reason": hold_why, "where": where,
                                                     "file": os.path.join(str(state_dir), DISPATCH_HOLD) if hold_kind == "operator" else None,
                                                     "identity": dict(ident), "wave_sha": spec["sha"],
                                                     "note": "the proxy is up and this endpoint has no completed drive, but no drive starts under a hold: settling and "
                                                             "probing go on; nothing is claimed, prepared for a start or spawned; said once per hold, again after it "
                                                             "lifts and returns; re-read at the start decision, before the durable claim and before the spawn"})
    return True


def petri_drive_pass(specs: Sequence[Mapping[str, str]], *, state_dir: Optional[str], repo: str, script: str, ssh_bin: str, dry_run: bool,
                     python: Optional[str] = None, bank_root: str = PETRI_GRADING_HOST_BANK_ROOT_DEFAULT, bundle_dir: str = PETRI_BUNDLE_DIR_DEFAULT,
                     roster: Optional[str] = None, env_allow: Sequence[str] = (), registry_root: Optional[str] = None, popen=subprocess.Popen,
                     run=subprocess.run) -> Dict[str, int]:
    """One pass over the specs (the module comment above), EVERYTHING about a tag under the global claims lock and its own tag lock: settle the live attempt; probe the
    endpoint; hold the proxy job to RUNNING; key the decision on the endpoint identity; prepare (checkout at the wave sha, interpreter,
    bundle, bank view); claim; spawn detached from the checkout (cwd, F22_WT, PYTHONPATH=<checkout>/src, F22_PY=the interpreter,
    F22_GRADING_HOST_BANK_ROOT=`bank_root` so the drive reads the view prepared here, SISTEER_SECRETS_MODE inherited or op, stdin /dev/null, a
    per-attempt log, the lock fd inherited until the pid marker is written; the
    environment an ALLOWLIST of the daemon's — `petri_drive_env` — so no selection knob reaches the drive); record `started` once the
    marker is there. The START is forbidden under the operator's `dispatch.hold` and under the schema hold (`petri_start_hold`,
    read right before the start decision; `registry_root` names the registry the schema is checked against, None = no schema check): settling
    and probing go on, one `petri_drive_held` per tag (said again after a release), the attempt is neither claimed nor recorded. Returns counts."""
    summary = {k: 0 for k in ("running", "completed", "incomplete", PETRI_ATTEMPT_UNATTESTED, "capped", "waiting", "stale", "started", "unreachable", "unprepared",
                              "locked", "pruned", "refused", "blocked", "held")}
    if specs and not state_dir:
        raise ValueError("--petri-drive / --petri-state need --state-dir (the drive's outcome record lives there)")
    python = python or sys.executable
    roster = roster or os.environ.get("F22_ROSTER") or PETRI_ROSTER_DEFAULT
    prep = dict(state_dir=state_dir, repo=repo, python=python, bank_root=bank_root, bundle_dir=bundle_dir, roster=roster, run=run)
    if specs:
        os.makedirs(state_dir, exist_ok=True)  # type: ignore[arg-type]
    for spec in specs:
        tag = spec["tag"]
        rec_path = petri_drive_record_path(state_dir, tag)  # type: ignore[arg-type]
        with _claims_lock(state_dir), _petri_lock(state_dir, tag) as lock_fd:  # type: ignore[arg-type]   # claims → tag (→ checkouts inside)
            if lock_fd is None:
                summary["locked"] += 1
                log_event(state_dir, {"event": "petri_drive_lock_held", "tag": tag, "note": "another process holds this tag's claim lock: skipped this pass"})
                continue
            rec = _petri_load(rec_path) or _petri_new_record(spec)
            live, changed = _petri_reconcile(rec, state_dir)
            if changed:
                _petri_save(rec_path, rec)
            if live is not None:
                summary["running"] += 1
                continue
            alive = _petri_survivors_block(rec, state_dir, tag)              # a settled attempt's legs still alive: stopped (TERM, then KILL) and no start
            if alive:                                                        # beside them, BEFORE any verdict (a capped endpoint's
                _petri_save(rec_path, rec)                                   # TERM-ignoring leg must still be killed, not left auditing under its locks)
                summary["blocked"] += 1
                _say_once(state_dir, ("survivors", tag, tuple(alive)), {"event": "petri_drive_previous_legs_alive", "tag": tag, "pids": alive,
                                                                        "note": "a previous attempt's legs (members of its session) are alive and hold the per-leg locks: "
                                                                                f"no new attempt until they are gone (TERM sent; KILL after {SURVIVOR_KILL_AFTER_S:g}s)"})
                continue
            if _spec_pass(spec) == PETRI_PASS_OVERFLOW:
                # the OVERFLOW pass starts only after the FIRST pass's drive settled `incomplete` for its bound job and none is live (petri-completion):
                # before that nothing is ledgered for the drive to select, or the two drives would share the stores
                ok, why = petri_overflow_gate(state_dir, spec)
                if not ok:
                    summary["waiting"] += 1
                    _say_once(state_dir, ("overflow_wait", tag, spec.get("first_job_id"), why), {"event": "petri_overflow_waiting_first_pass", "tag": tag, "base_tag": spec.get("base_tag"),
                                                                                              "first_job_id": spec.get("first_job_id"), "reason": why,
                                                                                              "note": "the overflow re-run drive starts once the first pass's record reads incomplete and no first-pass drive is live"})
                    continue
            status, mtime, content, detail = _petri_probe_endpoint(run, ssh_bin, spec["ssh"], spec["endpoint"])
            if status in ("unreachable", "error"):
                summary["unreachable"] += 1
                log_event(state_dir, {"event": "petri_drive_probe_failed", "tag": tag, "error": detail})
                continue
            if status == "absent":
                summary["waiting"] += 1
                _say_once(state_dir, ("waiting", tag, spec["sha"]), {"event": "petri_drive_waiting", "tag": tag, "endpoint": spec["endpoint"], "cluster": spec["cluster"], "wave_sha": spec["sha"]})
                if not dry_run and not _petri_completed_at_sha(rec, spec["sha"]):    # eager: the checkout, the venv link, the bundle, the bank
                    _petri_prepare_spec(spec, **prep)                               # view — ready before the proxy is; not after the wave's drive completed
                continue
            job_id = str((content or {}).get("slurm_job_id") or spec.get("job_id") or "")
            if not job_id:
                summary["unreachable"] += 1
                _say_once(state_dir, ("nojob", tag, mtime), {"event": "petri_drive_liveness_unknown", "tag": tag, "endpoint": spec["endpoint"],
                                                            "error": "endpoint.json names no slurm_job_id and the ledger binds no job: cannot hold the proxy to RUNNING"})
                continue
            state, detail = _petri_job_state(run, ssh_bin, spec["ssh"], job_id)
            if state in ("unreachable", "unknown"):
                summary["unreachable"] += 1
                _say_once(state_dir, ("liveness", tag, job_id, detail[:80]), {"event": "petri_drive_liveness_unknown", "tag": tag, "job_id": job_id, "error": detail})
                continue
            if state != "RUNNING":
                summary["stale"] += 1
                _say_once(state_dir, ("stale", tag, job_id, mtime), {"event": "petri_drive_stale_endpoint", "tag": tag, "job_id": job_id, "job_state": state,
                                                                     "ledger_job_id": spec.get("job_id") or None, "endpoint": spec["endpoint"], "endpoint_mtime": mtime,
                                                                     "note": "endpoint.json exists but the proxy job that wrote it is not RUNNING — not starting a drive against a dead server"})
                continue
            ident = _petri_identity(job_id, mtime or "", spec["sha"]); key = _identity_key(ident)
            verdict = _petri_verdict(rec, key)
            if verdict in ("completed", "incomplete", PETRI_ATTEMPT_UNATTESTED):
                summary[verdict] += 1
                if verdict == PETRI_ATTEMPT_UNATTESTED:
                    bad = [a for a in rec["attempts"] if a.get("state") == PETRI_ATTEMPT_UNATTESTED and _identity_key(a.get("identity")) == key]
                    _say_once(state_dir, ("unattested", tag, key), {"event": "petri_drive_unattested", "tag": tag, "identity": ident, "attempt": bad[-1].get("n") if bad else None,
                                                                    "log": bad[-1].get("log") if bad else None, "reason": bad[-1].get("reason") if bad else None,
                                                                    "note": "a federated drive's attestation was refused for this endpoint (round 1 on 7778694d0): no start until a "
                                                                            "human re-freezes the wave at a sha carrying the wif attestation (or fixes the org's ids file) and "
                                                                            "edits or deletes the record"})
                continue
            if verdict == "capped":
                summary["capped"] += 1
                logs = [a.get("log") for a in rec["attempts"] if _identity_key(a.get("identity")) == key]
                epc = (rec.get("endpoints") or {}).get(key) or {}
                uncovered = list(epc.get("remaining") or [])
                _say_once(state_dir, ("capped", tag, key), {"event": "petri_drive_gave_up", "tag": tag, "identity": ident, "attempts": len(logs), "logs": logs,
                                                            "n_uncovered": len(uncovered), "uncovered": uncovered, "covered": len(epc.get("covered") or []),
                                                            "required": len(set(epc.get("expected_ids") or []) - set(epc.get("skipped_runs") or [])),
                                                            "note": f"{PETRI_DRIVE_ATTEMPT_CAP} attempts failed or fell short against this endpoint; a human reads the logs and edits or deletes the record to retry"})
                continue
            if _petri_start_held(state_dir, registry_root, tag=tag, ident=ident, spec=spec, summary=summary, where="at the start decision"):
                continue
            overflow = _spec_pass(spec) == PETRI_PASS_OVERFLOW
            if dry_run:
                print(f"DRY-RUN: checkout {state_dir}/checkouts/{spec['sha']}; F22_WT=<checkout> F22_PY={python} PYTHONPATH=<checkout>/src "
                      f"F22_GRADING_HOST_BANK_ROOT={bank_root} " + (f"{PETRI_OVERFLOW_ENV}=1 " if overflow else "")
                      + f"SISTEER_SECRETS_MODE={os.environ.get('SISTEER_SECRETS_MODE', 'op')} bash <checkout>/{script} {spec['model']} {spec['view']} {spec['cluster']} {spec['sha']}"
                      + (f" {PETRI_OVERFLOW_LEGS_ARG}" if overflow else "")
                      + f" > {state_dir}/petri_drive.{tag}.<n>.log  (endpoint identity {key})", flush=True)
                summary["started"] += 1
                continue
            checkout = _petri_prepare_spec(spec, **prep)
            if checkout is None:
                summary["unprepared"] += 1
                continue
            script_path = script if os.path.isabs(script) else os.path.join(checkout, script)
            if not os.path.isfile(script_path):
                summary["unprepared"] += 1
                _say_once(state_dir, ("noscript", tag, spec["sha"], script_path), {"event": "petri_drive_unprepared", "tag": tag, "wave_sha": spec["sha"], "reason": f"no drive at {script_path}"})
                continue
            endpoints = rec.setdefault("endpoints", {})
            if key not in endpoints:   # the wave's EXPECTED run set under this proxy job, from its own registration — never from a drive
                ep, why = _petri_fetch_expected(run, ssh_bin, spec["ssh"], spec.get("shared") or _shared_of(spec["endpoint"]), job_id, (content or {}).get("cells_served"))
                if ep is None:
                    summary["unprepared"] += 1
                    _say_once(state_dir, ("expected", tag, key, why), {"event": "petri_drive_expected_unknown", "tag": tag, "identity": ident, "reason": why,
                                                                       "note": "no claim without the expected run set: completion could never be judged"})
                    continue
                ep["identity"] = ident
                endpoints[key] = ep
                _petri_save(rec_path, rec)
                log_event(state_dir, {"event": "petri_drive_expected", "tag": tag, "identity": ident, "registration": ep["registration"], "n_expected": len(ep["expected_ids"]),
                                      "skipped_objects": ep["skipped_objects"], "n_skipped_runs": len(ep["skipped_runs"])})
            try:                                                            # the allowlist + the VALIDATED operator overlay; never a selection knob;
                env, dropped, overlay_names = petri_drive_env(os.environ, env_allow, state_dir)   # read BEFORE the claim: a bad file refuses, records nothing
            except OverlayRefused as exc:
                summary["refused"] += 1
                _say_once(state_dir, ("overlay", tag, tuple(exc.reasons)), {"event": "petri_drive_overlay_refused", "tag": tag, "file": exc.file, "reasons": exc.reasons,
                                                                            "note": "a present but invalid overlay refuses every attempt until it is fixed or removed: "
                                                                                    "nothing recorded, nothing spawned, no partial application"})
                continue
            auditor = petri_expected_key_sha12(env, run=run, switched="F22_AUDITOR_ORG" in overlay_names)   # the hash the drive's DRIVE_START must attest
            if auditor.get("sha12") is None:
                summary["refused"] += 1
                kind = auditor.get("kind") or "unresolved"
                _say_once(state_dir, ("keyhash", tag, kind, str(auditor.get("error"))[:80]),
                          {"event": "petri_drive_key_mismatch" if kind == "mismatch" else "petri_drive_key_unresolved", "tag": tag, "org": auditor.get("org"),
                           "mode": auditor.get("mode"), "auth": auditor.get("auth") or "key", "error": auditor.get("error"),
                           "note": ("the inherited key is not the named org's vault item: the attested pool would be false; no attempt until the label or the key is fixed"
                                    if kind == "mismatch" else
                                    "the named org's federation ids file could not be read (absent, empty or incomplete): the drive would die on the same read; no attempt until it resolves"
                                    if auditor.get("auth") == "wif" else
                                    "the named org's key hash could not be resolved: the drive would die on the same fetch; no attempt until it resolves")})
                continue
            if _petri_start_held(state_dir, registry_root, tag=tag, ident=ident, spec=spec, summary=summary, where="before the durable claim"):   # the
                continue                                                        # preparation above took minutes; a hold that landed meanwhile stops the start here
            n = len(rec["attempts"]) + 1
            log_path = os.path.join(state_dir, f"petri_drive.{tag}.{n}.log")  # type: ignore[arg-type]
            argv = ["bash", script_path, spec["model"], spec["view"], spec["cluster"], spec["sha"]] + ([PETRI_OVERFLOW_LEGS_ARG] if overflow else [])
            att = {"n": n, "state": "claimed", "identity": ident, "pid": None, "proc_start": None, "log": log_path, "rc_file": log_path + ".rc",
                   "pid_file": log_path + ".pid", "result_file": log_path + ".result.json", "checkout": checkout, "argv": argv,
                   "claimed_utc": utc_now(), "started_utc": None, "exited_utc": None, "exit_code": None, "reason": None,
                   "auditor": auditor, "overlay": overlay_names, "key_attested": None, "pass": _spec_pass(spec)}
            if overflow:
                env[PETRI_OVERFLOW_ENV] = "1"      # the daemon's OWN setting from the spec, after the deny-list stripped any inherited copy: the drive's
                                                   # `ledgered` selector reads the legs itself (petri_run.py legs-status); the operator still selects nothing
            rec.update({"wave_sha": spec["sha"], "endpoint": spec["endpoint"], "ledger_job_id": spec.get("job_id") or None, "python": python})
            rec["attempts"].append(att)
            _petri_save(rec_path, rec)                                      # the DURABLE claim, before the spawn
            env.update({"F22_WT": checkout, "F22_PY": python, "PYTHONPATH": os.path.join(checkout, "src"), "PETRI_DRIVE_RC_FILE": att["rc_file"],
                        "PETRI_DRIVE_PID_FILE": att["pid_file"], "PETRI_DRIVE_RESULT_FILE": att["result_file"], "PETRI_LOCK_FD": str(lock_fd),
                        "F22_GRADING_HOST_BANK_ROOT": bank_root})   # the drive READS the view this daemon BUILT (--petri-bank-root), never its own literal
            att["env_keys"] = sorted(env); att["env_dropped"] = dropped   # names only, never values
            if _petri_start_held(state_dir, registry_root, tag=tag, ident=ident, spec=spec, summary=summary, where="before the spawn"):   # the claim
                rec["attempts"].pop()                                           # just written is WITHDRAWN -- it never spawned, it counts toward nothing (no `claimed`
                _petri_save(rec_path, rec)                                      # attempt for the reconciler to settle as failed); the record reads as before it
                continue
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    proc = popen(["bash", "-c", PETRI_WRAPPER, *argv[1:]], cwd=checkout, env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=(lock_fd,))
            except (OSError, subprocess.SubprocessError) as exc:
                att.update({"state": "failed", "exited_utc": utc_now(), "reason": f"spawn failed: {type(exc).__name__}: {exc}"})
                _petri_save(rec_path, rec)
                summary["unprepared"] += 1
                log_event(state_dir, {"event": "petri_drive_exited", "tag": tag, "attempt": n, "pid": None, "exit_code": None, "state": "failed", "log": log_path, "reason": att["reason"]})
                continue
            _PETRI_PROCS[tag] = proc
            marker = _wait_marker(att["pid_file"], proc)                    # the wrapper's first action; the lock is held (its fd inherited) until it is written
            pid = getattr(proc, "pid", None)
            att.update({"state": "started", "pid": pid, "proc_start": (marker or {}).get("proc_start") or (_proc_start(pid) if isinstance(pid, int) else None),
                        "started_utc": utc_now(), "marker": marker is not None})
            _petri_save(rec_path, rec)
            summary["started"] += 1
            log_event(state_dir, {"event": "petri_drive_started", "tag": tag, "attempt": n, "pid": pid, "log": log_path, "checkout": checkout, "wave_sha": spec["sha"],
                                  "identity": ident, "ledger_job_id": spec.get("job_id") or None, "python": python, "argv": " ".join(shlex.quote(a) for a in argv),
                                  "pass": _spec_pass(spec), "legs": PETRI_OVERFLOW_LEGS_ARG if overflow else "ALL",
                                  "marker": marker is not None, "env_dropped": dropped, "overlay": overlay_names,
                                  "auditor_org": auditor.get("org"), "secrets_mode": auditor.get("mode"), "auditor_auth": auditor.get("auth") or "key",
                                  "auditor_key_sha12_expected": auditor.get("sha12") if auditor.get("auth") != "wif" else None,      # a key drive's: the vault item's hash
                                  "auditor_ids_sha12_expected": auditor.get("sha12") if auditor.get("auth") == "wif" else None,      # a federated drive's: the ids file's digest
                                  "note": None if not spec.get("job_id") or spec.get("job_id") == job_id else
                                  f"the ledger binds job {spec.get('job_id')} but endpoint.json names {job_id} (RUNNING): driving the job that wrote the file"})
    if specs and state_dir and not dry_run:
        summary["pruned"] = len(prune_petri_checkouts(state_dir=state_dir, repo=repo, specs=specs, run=run))
    return summary


def petri_pass_args(args, petri_specs: Sequence[Mapping[str, str]], petri_clusters: Mapping[str, Mapping[str, str]]) -> Dict[str, int]:
    """Discovery + one pass, from the parsed arguments (the thread's body and `--once`'s inline pass)."""
    specs_now = list(petri_specs) + discover_petri_specs(args.petri_state, args.petri_model, petri_clusters, args.state_dir)
    return petri_drive_pass(specs_now, state_dir=args.state_dir, repo=args.repo, script=args.petri_drive_script, ssh_bin=args.ssh_bin, dry_run=args.dry_run,
                            python=args.petri_python or args.python, bank_root=args.petri_bank_root, bundle_dir=args.petri_bundle_dir,
                            env_allow=tuple(args.petri_drive_env_allow or ()), registry_root=args.registry_root)   # the schema hold covers the start


def petri_loop(args, petri_specs: Sequence[Mapping[str, str]], petri_clusters: Mapping[str, Mapping[str, str]], stop: threading.Event) -> None:
    """The Petri pass on its own cadence (`--petri-interval`, floored at PETRI_INTERVAL_FLOOR_S), independent of the grading loop's
    dispatch: the first pass runs at once, a failing pass is one `petri_pass_failed` event and never ends the loop."""
    while True:
        try:
            petri_pass_args(args, petri_specs, petri_clusters)
        except Exception as exc:  # noqa: BLE001 — a pass never kills the watch
            log_event(args.state_dir, {"event": "petri_pass_failed", "error": f"{type(exc).__name__}: {exc}"})
        if stop.wait(max(float(args.petri_interval), PETRI_INTERVAL_FLOOR_S)):
            return


def coordination_line(args) -> str:
    """The ready-to-post line for the site's coordination channel (SISTEER_COORDINATION_CHANNEL), or "" when the site
    names none (then nothing is printed)."""
    ch = coordination.channel()
    if not ch:
        return ""
    return (f"coordination (post to {ch}; the daemon does not wait for it): "
            f"{args.concurrency} haiku+sonnet lowprio {args.expected_hours}h — grading daemon for {', '.join(args.model_id)} "
            f"({'once' if args.once else f'every {args.interval} s'})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry-root", default=os.environ.get(RW.ROOT_ENV) or DB.DEFAULT_ROOT)
    ap.add_argument("--model-id", action="append", required=True, help="the model ids whose pending runs this daemon grades (repeatable)")
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--families", default=None, help="comma list of families to grade (default: every family)")
    ap.add_argument("--once", action="store_true", help="one tick, then exit")
    ap.add_argument("--interval", type=float, default=120.0, help="seconds between ticks (default 120; floored at 30)")
    ap.add_argument("--max-ticks", type=int, default=0, help="stop after this many ticks (0 = run until stopped)")
    ap.add_argument("--store-root-map", action="append", default=None, metavar="SUBSTRATE=REMOTE_ROOT=LOCAL_ROOT")
    ap.add_argument("--mirror", action="append", default=None, metavar="SUBSTRATE=SSH_ALIAS", help="rsync each group's stores from that cluster before dispatch")
    ap.add_argument("--rsync-bin", default="rsync")
    ap.add_argument("--mirror-min-free-bytes", type=int, default=MIRROR_MIN_FREE_BYTES_DEFAULT,
                    help="the grade mirror's free-space floor (D160): below it a group's mirror + dispatch is skipped for the tick (default 20 GiB)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--configs-root", default=os.path.join(REPO, "configs"),
                    help="where the drivers' configs live (config_for: <root>/<private_run>.yaml, <root>/sdfbase/static_padding/..., "
                         "<root>/capeval/manifests/...); default <repo>/configs")
    ap.add_argument("--grades-root", default=None, help="the graders' output root (--grades-root of the static / padded / multi-turn / BoN drivers); "
                                                        "a family whose template needs it is refused by name without it")
    ap.add_argument("--driver", action="append", default=None, metavar="FAMILY=TEMPLATE", help="override a family's command template")
    ap.add_argument("--extra", action="append", default=None, metavar="FAMILY=ARGS", help="further arguments for the family's driver (appended)")
    ap.add_argument("--eval-family", action="append", default=None, metavar="EVAL_ID=FAMILY", help="route an eval id the built-in table does not name to a family")
    ap.add_argument("--no-contrasts", dest="contrasts", action="store_false", help="skip the run_contrast pass")
    ap.add_argument("--no-contrasts-sweep", dest="contrasts_sweep", action="store_false",
                    help="g28-fix-3 / sweephold / sweephold-fix-1 / -fix-2: the INITIAL state of the sweep hold only, applied ONCE per state dir -- the first "
                         "start with this flag writes <state-dir>/contrasts_sweep_hold (reason: argv) when absent and records it in contrasts_sweep_hold.seeded; "
                         "every later start or code-drift re-exec with the flag finds the record and writes nothing, so a released hold stays released. "
                         "With no record but evidence of an earlier sweephold-era daemon in the state dir (its event log / mark) NOTHING is written and "
                         "daemon_started says skipped: ambiguous_legacy_state -- the operator then touches the hold file to hold, or writes "
                         "contrasts_sweep_hold.seeded (any JSON object) to acknowledge the released state. From then on the file alone holds or releases the "
                         "once-per-code-advance re-emission sweep, tick by tick: a hold is a file touch, a release an rm (the first line is the reason) -- the flag "
                         "is never re-consulted (a --dry-run daemon writes nothing and holds for itself). sweephold-fix-5 / -fix-6: over history nobody can "
                         "classify the flag's hold is PENDING the operator's decision -- persisted in <state-dir>/contrasts_sweep_pending_decision and holding the "
                         "sweep on every flagged start until the hold file or the .seeded record exists")
    ap.add_argument("--spool", default=None, help="the spool emit_contrasts writes to (default: the emitter's resolution order)")
    ap.add_argument("--state-dir", default=None, help="where grade_daemon.jsonl goes (default: no file, stdout only)")
    ap.add_argument("--driver-timeout-s", type=float, default=6 * 3600)
    ap.add_argument("--grader-session", default=None,
                    help="the worklist session the graders this daemon launches claim under (default worklist.default_session(): the session manager's session name, "
                         "else $USER@host — the drivers' own default); a row that session holds is re-dispatched, another session's is left alone")
    ap.add_argument("--grader-lock-timeout-s", type=float, default=GRADER_LOCK_TIMEOUT_S_DEFAULT,
                    help=f"how long a grader's registry claim waits for the harvester's write lock ({GRADER_LOCK_TIMEOUT_ENV} in the grader's env "
                         "unless already set; must outlast a harvester cycle)")
    ap.add_argument("--concurrency", type=int, default=200, help="for the coordination line")
    ap.add_argument("--expected-hours", type=int, default=24, help="for the coordination line")
    ap.add_argument("--petri-drive", action="append", default=None, metavar="tag=..;model=..;view=..;cluster=..;ssh=..;endpoint=..;sha=..",
                    help="an explicit Petri proxy whose drive this daemon starts once its endpoint.json lands (D121(B)); repeatable; beside --petri-state")
    ap.add_argument("--petri-state", default=None, help="the model's front-door state dir (WAVE.<view>.json + submissions.jsonl + attempts.jsonl): the drive specs are re-discovered from it EVERY pass; "
                                                         "each unit's drive checks out the sha ITS submission was made at (the row's extra.sha), never the wave file's")
    ap.add_argument("--petri-model", default=None, help="the front-door model key the state dir belongs to")
    ap.add_argument("--petri-cluster", action="append", default=None, metavar="LABEL=SSH_ALIAS=WAVE_ROOT", help="a cluster's ssh alias + wave root, for the discovered specs; repeatable")
    ap.add_argument("--petri-interval", type=float, default=PETRI_INTERVAL_DEFAULT_S, help=f"seconds between Petri passes (its own thread; default {PETRI_INTERVAL_DEFAULT_S:g}, floored at {PETRI_INTERVAL_FLOOR_S:g})")
    ap.add_argument("--petri-drive-script", default=PETRI_DRIVE_SCRIPT_DEFAULT, help="the drive, relative to the checkout at the wave sha (a fake under test)")
    ap.add_argument("--petri-python", default=None, help="the interpreter the drive runs under (F22_PY; must import inspect_ai + petri_bloom); default --python")
    ap.add_argument("--petri-bundle-dir", default=PETRI_BUNDLE_DIR_DEFAULT, help="the frozen Petri scenario bundle the drive reads")
    ap.add_argument("--petri-bank-root", default=PETRI_GRADING_HOST_BANK_ROOT_DEFAULT, help="the grading host bank-view root the drive reads (<root>/<model>/cells_view/<view>); built there when absent and "
                                                                                       "handed to the drive as F22_GRADING_HOST_BANK_ROOT (08_grade.sh passes $SISTEER_OBJECTS_BANK, the front door's staged bank)")
    ap.add_argument("--petri-drive-env-allow", action="append", default=None, metavar="NAME|PREFIX*",
                    help="a further variable (or PREFIX*) of the daemon's environment the drive may inherit; the selection knobs are never inherited")
    ap.add_argument("--ssh-bin", default="ssh", help="the ssh the endpoint probe uses (a fake under test)")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.families = [f.strip() for f in args.families.split(",") if f.strip()] if args.families else None
    templates = dict(DEFAULT_TEMPLATES)
    templates.update(_kv(args.driver, "--driver"))
    extra = _kv(args.extra, "--extra")
    mirror_ssh = _kv(args.mirror, "--mirror")
    extra_families = _kv(args.eval_family, "--eval-family")
    try:
        store_root_map = P.parse_store_root_map(args.store_root_map) or None
        petri_specs = parse_petri_specs(args.petri_drive)
        petri_clusters = parse_petri_clusters(args.petri_cluster)
        petri_watch = bool(petri_specs or (args.petri_state and args.petri_model))
        if petri_watch and not args.state_dir:
            raise ValueError("--petri-drive / --petri-state need --state-dir (the drive's outcome record lives there)")
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if not os.path.exists(DB.db_path(args.registry_root)):
        print(f"REFUSED: no registry database under {args.registry_root}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.configs_root):
        print(f"REFUSED: --configs-root {args.configs_root} is not a directory", file=sys.stderr)
        return 2
    coord = coordination_line(args)
    if coord:
        print(coord, flush=True)
    # the Petri pass on its own thread: a judge subprocess may hold `tick` for hours; the proxy's endpoint must be acted on
    # within minutes. `--once` runs the single pass inline after its tick instead.
    sweep_seeding = contrasts_sweep_hold_initial(args)       # sweephold / sweephold-fix-1: argv --no-contrasts-sweep seeds the hold FILE once per state dir (persisted); the file decides from here
    log_event(args.state_dir, {"event": "daemon_started", "daemon_sha": IMPORTED_CODE_SHA, "pid": os.getpid(), "once": bool(args.once), "dry_run": bool(args.dry_run),
                               "contrasts_sweep_hold_seeded": sweep_seeding.get("seeded_now"), "contrasts_sweep_hold_seeding": sweep_seeding})
    # the schema preflight BEFORE either loop starts -- a `schema_hold` lands here, ahead of the first Petri pass and the first tick
    # (both re-check on their own and neither re-says a gap already said); held or not, the loops start and keep checking
    SCHEMA_GATE.check(args.registry_root, args.state_dir)
    stop = threading.Event(); petri_thread: Optional[threading.Thread] = None
    if petri_watch and not args.once:
        petri_thread = threading.Thread(target=petri_loop, args=(args, petri_specs, petri_clusters, stop), name="petri-drive", daemon=True)
        petri_thread.start()
    n = 0
    try:
        while True:
            n += 1
            if check_code_drift(args) and not args.once:
                stop.set()
                if petri_thread is not None:
                    petri_thread.join(timeout=PETRI_JOIN_TIMEOUT_S)
                reexec_self()
            tick(args, templates, extra, mirror_ssh, store_root_map, extra_families)
            if petri_watch and args.once:
                petri_pass_args(args, petri_specs, petri_clusters)
            if args.once or (args.max_ticks and n >= args.max_ticks):
                return 0
            time.sleep(max(args.interval, TICK_INTERVAL_FLOOR_S))
    finally:
        stop.set()
        if petri_thread is not None:
            petri_thread.join(timeout=PETRI_JOIN_TIMEOUT_S)


if __name__ == "__main__":
    raise SystemExit(main())
