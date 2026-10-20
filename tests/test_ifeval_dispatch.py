"""The IFEval battery's thin wave-wide dispatcher (``scripts/tinycap_ifeval_dispatch.py``)
on a SYNTHETIC 40-row matrix of the frozen matrix's shape (30 launch + 2 held + 8 deferred):
seven slugs (the nine-cell ``gemma4e4b-sdfbase-main`` view, eight
``nemotron3nano4b-sdfbase-main`` rows on the Nemotron image, four held rows),
the cap-keyed path tables spelled as JSON spells them (string keys).

Covers (P39): the aggregate outstanding peak is EXACTLY ``MAX_GPU_CONTAINERS`` (8)
across both image functions and never 9, measured by the fake FunctionCall
registry; queued / active / done are disjoint in the state file at intermediate
ticks; a failing call releases its slot; a refusal REC is filed REFUSED and never
re-spawned; a restart adopts the active set by FunctionCall id and re-spawns nothing
whose store validates; held rows never enter the queue; every spawn goes through
``launch_configurations`` with one configuration and ``wait=False``; ``--dry-run``
spawns nothing; ``--stop`` writes the sentinel and runs ``modal app stop <app>
--yes``; the GPU kwargs name a lens row's ``object_sha256`` and none for a base row;
a 4096-rung row's planning inputs (``off_cap_by_task`` / ``item_subset`` /
``out_dir`` / ``bank_view``). The launcher is a stub module (the real one needs the
Modal stub loader); the contracts are real. Pure: no Modal, no network."""
from __future__ import annotations

import collections
import contextlib
import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tinycap_ifeval_dispatch.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tinycap_ifeval_dispatch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


D = _load_script()

SHA = "0123456789abcdef0123456789abcdef01234567"
SHA7 = SHA[:7]
WAVE_ROOT = f"{C.VOL_TINYCAP_ROOT}/ifeval_canonical/{SHA7}"
APP = f"ifeval-canonical-{SHA7}"
CAPS = (2048, 4096, 8192)
NEMOTRON_FAMILY = "nemotron_h"
IT_CELLS = ["none", "prompted__safety_refusal", "prompted__si_constitution", "constv2-mix-full__lora_r4_ce"]
MAIN_CELLS = ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]
NEMO_CELLS = ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "prompted__safety_refusal",
              "prompted__si_constitution", "constv2-mix-full__lora_r4_ce_minus_ctl"]
LLAMA_CELLS = ["none", "prompted__safety_refusal", "prompted__si_constitution", "constv2-mix-full__lora_r4_ce",
               "constv2ctl2-full__lora_r4_ce"]
HELD_CELLS = {"constv2-mix-full__lora_r4_ce"}
HELD_REASON = "the instruct-direct constitution objects are being re-fitted; a hard exclusion until the refit lands"
#: (model key, view, cells): 4 + 4 + 4 + 4 + 6 + 5 + 5 = 32 launch/held rows + 8 DEFERRED slots = 40 rows; the
#: gemma / qwen `it` constitution LoRA rows are held (2), as in the frozen matrix
VIEWS = [("gemma4e4b", "it", IT_CELLS), ("gemma4e4b", "main", MAIN_CELLS), ("qwen35_4b", "it", IT_CELLS),
         ("qwen35_4b", "main", MAIN_CELLS), ("nemotron3nano4b", "main", NEMO_CELLS),
         ("llama8b", "main", LLAMA_CELLS), ("llama8b", "it", LLAMA_CELLS)]
HELD_MODEL_KEYS = {"gemma4e4b", "qwen35_4b"}
#: the deferred design slots (AlphaSteer / Arditi at minimal over-refusal strength), one per (arm, model), LAST
DEFERRED = [(arm, key, view) for arm in ("alphasteer", "arditi")
            for key, view in (("gemma4e4b", "it"), ("qwen35_4b", "it"), ("nemotron3nano4b", "main"), ("llama8b", "main"))]


def make_row(index: int, model_key: str, view: str, cell: str, *, held: bool, deferred: bool = False) -> dict:
    tag = model_key if view == "main" else f"{model_key}_{view}"
    slug = C.IFEVAL_SLUGS[(model_key, view)]
    row = C.model_row(slug)
    cell_slug = cell.replace("__", "--")
    fitted = cell != "none" and not cell.startswith("prompted__")
    return {"index": index, "job_id": f"ife-{model_key}-{view}-{cell_slug}", "model_key": model_key, "view": view,
            "tag": tag, "slug": slug, "cell": cell, "cell_slug": cell_slug, "fitted": fitted,
            "object_sha256": (hashlib.sha256(f"{tag}|{cell}".encode()).hexdigest() if fitted and not deferred else None),
            "route_kind": "lens" if fitted else ("prompted" if cell.startswith("prompted__") else "base"), "thinking": C.THINKING_OFF,
            "sampling_mode": C.SAMPLING_VENDOR_PRESET, "samples_per_item": C.IFEVAL_SAMPLES_PER_ITEM,
            "cap": CAPS[0], "caps": list(CAPS), "cap_ceiling": CAPS[-1],
            "geometry_eval_by_cap": {str(c): (f"capability:ifeval_longsi_{c}" if cell == "prompted__si_constitution"
                                              else f"capability:ifeval_{c}") for c in CAPS},
            "out_root": f"controls/{tag}/{cell_slug}/vendor_preset",
            "rung_out_roots": {str(c): f"controls/{tag}/{cell_slug}/vendor_preset__rung{c}" for c in CAPS[1:]},
            "subset_paths": {str(c): f"rung{c}/{tag}/{cell_slug}/rung_subset.json" for c in CAPS[1:]},
            "dispatch_paths": {str(c): f"dispatch/{tag}/{cell_slug}/rung{c}.json" for c in CAPS},
            "configuration_ids": {str(c): f"{tag}/{cell_slug}/rung{c}" for c in CAPS},
            "family": row.family, "image": "nemotron" if row.family == NEMOTRON_FAMILY else "default",
            "status": "deferred" if deferred else "held" if held else "launch", "held_reason": HELD_REASON if held else None,
            "deferred": ({"arm": cell, "record": f"configs/sdfbase/ifeval/deferred/{cell}__{model_key}.json",
                          "record_fields": ["object_sha256", "view", "rung", "selection_rule"], "filled": False}
                         if deferred else None)}


def make_matrix() -> dict:
    rows: List[dict] = []
    for model_key, view, cells in VIEWS:
        for cell in cells:
            held = view == "it" and model_key in HELD_MODEL_KEYS and cell in HELD_CELLS
            rows.append(make_row(len(rows), model_key, view, cell, held=held))
    for arm, model_key, view in DEFERRED:
        rows.append(make_row(len(rows), model_key, view, arm, held=False, deferred=True))
    return {"schema": "ifeval_matrix/1", "experiment": "ifeval-canonical-battery", "rows": rows}


MATRIX = make_matrix()
LAUNCH = [r for r in MATRIX["rows"] if r["status"] == "launch"]
HELD = [r for r in MATRIX["rows"] if r["status"] == "held"]
DEFERRED_ROWS = [r for r in MATRIX["rows"] if r["status"] == "deferred"]
CID = {(r["slug"], r["cell"]): r["configuration_ids"]["2048"] for r in MATRIX["rows"]}
ROW_BY_CID = {r["configuration_ids"]["2048"]: r for r in MATRIX["rows"]}


# --------------------------------------------------------------------------- #
# fakes: FunctionCalls whose completion the test controls, and a stub launcher #
# --------------------------------------------------------------------------- #
class FakeCall:
    def __init__(self, fc_id: str, fn_name: str, cfg: dict, kwargs: dict):
        self.object_id, self.fn_name, self.cfg, self.kwargs = fc_id, fn_name, cfg, kwargs
        self.finished, self._rec, self._error = False, None, None

    def get(self, timeout=None):
        if not self.finished:
            raise TimeoutError("still running")
        if self._error is not None:
            raise self._error
        return self._rec


def remote_result(exc: BaseException) -> BaseException:
    """Mark ``exc`` the way Modal's ``_process_result`` marks a deserialised REMOTE exception (``append_modal_tb``): its
    traceback starts at a frame whose filename is ``<ta-…>:<path>`` and it gains ``__line_cache__``. This is the
    affirmative evidence the dispatcher's status boundary requires before it files a call as finished (terminal requires
    remote evidence); an unmarked exception out of ``get`` is a status NOT obtained, never the call's failure."""
    try:
        exec(compile("raise exc", f"{D.REMOTE_TRACEBACK_FRAME_PREFIX}01SYNTHETICTASKID000000000>:/root/driver.py", "exec"),
             {"exc": exc})
    except BaseException as raised:  # noqa: BLE001 — the same object, now carrying the remote frame
        assert raised is exc
    setattr(exc, D.REMOTE_TRACEBACK_ATTR, {})
    return exc


class FakeRegistry:
    """Every spawn across both functions; ``outstanding`` = spawned and not yet
    finished, ``peak`` its maximum — the aggregate the dispatcher must cap at 8."""

    def __init__(self):
        self.calls: Dict[str, FakeCall] = {}
        self.order: List[str] = []
        self.outstanding = self.peak = 0
        self.spawns_by_cid: collections.Counter = collections.Counter()

    def spawn(self, fn_name: str, cfg: dict, kwargs: dict) -> FakeCall:
        fc = FakeCall(f"fc-{len(self.order):03d}", fn_name, cfg, kwargs)
        self.calls[fc.object_id] = fc
        self.order.append(fc.object_id)
        self.outstanding += 1
        self.peak = max(self.peak, self.outstanding)
        self.spawns_by_cid[cfg["configuration_id"]] += 1
        return fc

    def unfinished(self) -> List[FakeCall]:
        return [self.calls[i] for i in self.order if not self.calls[i].finished]

    def finish(self, fc: FakeCall, rec: Optional[dict] = None, error: Optional[BaseException] = None) -> None:
        """The call finished AT MODAL: a REC, or the remote exception handed back as the call's result — marked the way
        Modal marks it (``remote_result``). A raw, unmarked exception out of ``get`` is a status NOT obtained; the
        status-boundary tests inject those directly, without finishing the call here."""
        assert not fc.finished
        fc.finished, fc._rec, fc._error = True, rec, (remote_result(error) if error is not None else None)
        self.outstanding -= 1


class FakeFunction:
    def __init__(self, registry: FakeRegistry, name: str):
        self.registry, self.__name__ = registry, name

    def spawn(self, cfg: dict, **kwargs):
        return self.registry.spawn(self.__name__, cfg, kwargs)


def make_launcher(registry: FakeRegistry) -> types.ModuleType:
    L = types.ModuleType("tinycap_served_stub")
    L.REC_OK_STATUSES = ("OK", "DONE", "PARTIAL_COMPLETE")
    L.SPAWNED_LOG_DEFAULT = "spawned.jsonl"
    L.plan_calls, L.launch_calls, L.budget_calls = [], [], []
    L.run_configuration = FakeFunction(registry, "run_configuration")
    L.run_configuration_nemotron = FakeFunction(registry, "run_configuration_nemotron")

    def plan_configurations(cells=None, modes=None, controls=None, *, in_depth=False, model=C.MODEL_SLUG,
                            allow_out_of_support_thinking=False, sampling_mode=C.SAMPLING_DEFAULT,
                            samples=C.MATH500_SAMPLES_PER_ITEM, allow_off_cap_override=False,
                            reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, pass_no=1,
                            off_cap_by_task=None, item_subset=None, bank_view=None, out_dir=None, **extra):
        # the real planner takes `serving_path` and marks a NATIVE configuration with the key (present iff
        # native; a lens plan carries none) — the stub mirrors that, and records the kwarg so a test can pin whether the
        # dispatcher passed it at all (an unstamped row passes nothing: byte-identical plan calls)
        assert set(extra) <= {"serving_path"}, extra
        row = C.model_row(model)
        L.plan_calls.append(dict(cells=list(cells), modes=list(modes), controls=list(controls), model=model,
                                 sampling_mode=sampling_mode, samples=samples, reduction_mode=reduction_mode,
                                 pass_no=pass_no, off_cap_by_task=off_cap_by_task, item_subset=item_subset,
                                 bank_view=bank_view, out_dir=out_dir, **extra))
        native = extra.get("serving_path") == "lora_native"
        return [{"name": C.container_name(cell, mode, model_slug=row.slug), "model": row.slug, "cell": cell,
                 "mode": mode, "controls": list(controls), "steered": cell != "none", "base": cell == "none",
                 "out_dir": out_dir, "item_subset": item_subset, "off_cap_by_task": off_cap_by_task,
                 "bank_view": bank_view, **({"serving_path": "lora_native"} if native else {})} for cell in cells for mode in modes]

    def configuration_function(row):
        return L.run_configuration_nemotron if row.family == NEMOTRON_FAMILY else L.run_configuration

    def gpu_container_budget(n, *, allow_extra=False):
        L.budget_calls.append(n)
        if n > C.MAX_GPU_CONTAINERS and not allow_extra:
            raise SystemExit(f"refusing {n} GPU containers")

    def journal(spawned_log, cfg, fc_id):
        """The real launcher's `spawned.jsonl` line (name / model / cell / fc_id / t + configuration_id / launch_key), so
        the dispatcher's reservation reconciliation is exercised against the journal it reads in production."""
        if spawned_log is None:
            return
        p = Path(spawned_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"name": cfg.get("name"), "model": cfg.get("model"), "cell": cfg.get("cell"), "fc_id": fc_id,
                                 "t": 0.0, "configuration_id": cfg.get("configuration_id"), "launch_key": cfg.get("launch_key")}) + "\n")

    def launch_configurations(fn, cfgs, kwargs, *, spawned_log=None, wait=True):
        gpu_container_budget(len(cfgs))
        L.launch_calls.append({"fn": fn.__name__, "n": len(cfgs), "wait": wait, "spawned_log": spawned_log})
        out = []
        for cfg in cfgs:
            fc = fn.spawn(dict(cfg), **dict(kwargs))
            if L.crash_before_journal:
                raise RuntimeError("dispatcher killed after Modal accepted the spawn, before the journal line")
            journal(spawned_log, cfg, getattr(fc, "object_id", None))
            if L.crash_after_journal:
                raise RuntimeError("dispatcher killed after the journal line, before the state transition")
            out.append((cfg, fc))
        return out

    L.crash_before_journal = L.crash_after_journal = False
    L.journal = journal

    def app_name_command(name):
        return f"{C.MODAL_APP_NAME_ENV}={name}"

    L.launch_argv_calls = []

    def launch_argv(sha, *, model=C.MODEL_SLUG, app_name="", cells="", modes="", controls="",
                    samples=C.MATH500_SAMPLES_PER_ITEM, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                    off_cap_by_task="", serving_path=None, **_):
        # the real composer spells --serving-path whenever it is given and differs from the entry-point default; the stub
        # spells it whenever GIVEN (and records the kwarg), so a test can pin what the dispatcher handed over
        L.launch_argv_calls.append({"model": model, "cells": cells, "serving_path": serving_path})
        return ([app_name_command(app_name), "modal", "run", "scripts/modal/tinycap_served.py", "--sha", sha,
                 "--app-name", app_name, "--model", model, "--reduction-mode", reduction_mode, "--cells", cells,
                 "--modes", modes, "--controls", controls, "--samples", str(samples)]
                + ([C.FLAG_OFF_CAP_BY_TASK, off_cap_by_task] if off_cap_by_task else [])
                + (["--serving-path", str(serving_path)] if serving_path else []))

    for f in (plan_configurations, configuration_function, gpu_container_budget, launch_configurations,
              app_name_command, launch_argv):
        setattr(L, f.__name__, f)
    return L


class FakeClock:
    def __init__(self):
        self.t = 1_000.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class Wave:
    """One dispatcher over the synthetic matrix with a controllable world: ``sleep``
    is where the test finishes calls (``on_tick`` first, then ``complete_per_tick``
    oldest unfinished calls with an OK REC) and snapshots the state file."""

    def __init__(self, tmp_path: Path, registry: Optional[FakeRegistry] = None, *, complete_per_tick: int = 3,
                 on_tick=None, store_complete=None, adopt=None, max_ticks: int = 500, **kw):
        self.registry = registry or FakeRegistry()
        self.launcher = make_launcher(self.registry)
        self.local_root = tmp_path / "local"
        self.state_path = self.local_root / D.IFEVAL_DISPATCH_STATE
        self.snapshots: List[dict] = []
        self.log: List[str] = []
        self.complete_per_tick, self.on_tick = complete_per_tick, on_tick

        def sleep(_s):
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.snapshots.append(st)
            if self.on_tick:
                self.on_tick(self, st, len(self.snapshots))
            for fc in self.registry.unfinished()[:self.complete_per_tick]:
                self.registry.finish(fc, rec={"status": "OK", "name": fc.cfg["name"], "cell": fc.cfg["cell"]})

        self.d = D.Dispatcher(self.launcher, MATRIX, sha=SHA, app_name=APP, wave_root=WAVE_ROOT,
                              local_root=self.local_root, state_path=self.state_path, sleep=sleep, clock=FakeClock(),
                              adopt=adopt or (lambda fc_id: self.registry.calls[fc_id]),
                              store_complete=store_complete or (lambda cfg: False), max_ticks=max_ticks,
                              log=self.log.append, **kw)

    def run(self) -> int:
        return self.d.run()

    def state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# pins + the matrix fixture's shape                                            #
# --------------------------------------------------------------------------- #
def test_pins_and_fixture_shape():
    assert C.MAX_GPU_CONTAINERS == 8
    assert len(set(C.IFEVAL_SLUGS.values())) == 13            # 7 instrument-view slugs + the 6 seed lane views
    assert len(MATRIX["rows"]) == 40 and len(LAUNCH) == 30 and len(HELD) == 2 and len(DEFERRED_ROWS) == 8
    by_slug = collections.Counter(r["slug"] for r in MATRIX["rows"])
    assert by_slug["gemma4e4b-sdfbase-main"] == 4 and by_slug["nemotron3nano4b-sdfbase-main"] == 8
    # the synthetic matrix has the frozen matrix's shape (indices, statuses, cells per slug)
    frozen = json.loads((REPO / "configs/sdfbase/ifeval/matrix.json").read_text(encoding="utf-8"))["rows"]
    assert [(r["slug"], r["cell"], r["status"]) for r in frozen] == [(r["slug"], r["cell"], r["status"]) for r in MATRIX["rows"]]
    assert all(r["image"] == "nemotron" for r in MATRIX["rows"] if r["slug"] == "nemotron3nano4b-sdfbase-main")
    assert all(r["held_reason"] for r in HELD) and {r["view"] for r in HELD} == {"it"}
    assert C.is_refusal_status(C.REC_NEMO_BACKEND_UNATTESTED) and not C.is_refusal_status("OK")
    assert D.IFEVAL_DISPATCH_STATE == "dispatch_state.json"


# --------------------------------------------------------------------------- #
# (a) (b) (f) (g): the full wave                                               #
# --------------------------------------------------------------------------- #
def test_full_wave_peak_exactly_eight_across_both_functions(tmp_path):
    w = Wave(tmp_path)
    rc = w.run()
    assert rc == D.EXIT_OK
    reg = w.registry
    assert reg.peak == 8, "the aggregate outstanding peak must be exactly MAX_GPU_CONTAINERS"
    assert len(reg.order) == 30 and set(reg.spawns_by_cid.values()) == {1}
    assert collections.Counter(reg.calls[i].fn_name for i in reg.order) == \
        {"run_configuration": 24, "run_configuration_nemotron": 6}
    # (g) every spawn through launch_configurations with ONE configuration, wait=False
    assert len(w.launcher.launch_calls) == 30
    assert all(c["n"] == 1 and c["wait"] is False for c in w.launcher.launch_calls)
    assert max(w.launcher.budget_calls) == 1
    assert all(Path(c["spawned_log"]) == w.local_root / "spawned.jsonl" for c in w.launcher.launch_calls)
    # (b) queued / active / done disjoint in the state file at every intermediate tick, all three seen at once
    saw_all_three = False
    for st in w.snapshots:
        q, a, dn = set(st["queued"]), set(st["active"]), set(st["done"])
        assert not (q & a) and not (q & dn) and not (a & dn)
        assert len(a) <= 8
        saw_all_three = saw_all_three or (q and a and dn)
    assert saw_all_three
    assert w.snapshots[0]["active"] and len(w.snapshots[0]["active"]) == 8 and len(w.snapshots[0]["queued"]) == 30 - 8
    # (f) held rows never enter the queue; they are listed as held with their reason — and the DEFERRED slots
    # (the follow-up) never enter either, listed beside them with their record as the reason
    final = w.state()
    held_ids = {r["configuration_ids"]["2048"] for r in HELD}
    deferred_ids = {r["configuration_ids"]["2048"] for r in DEFERRED_ROWS}
    assert set(final["held"]) == held_ids | deferred_ids and all(final["held"][h] == HELD_REASON for h in held_ids)
    assert all(final["held"][d].startswith("DEFERRED:") and "configs/sdfbase/ifeval/deferred/" in final["held"][d] for d in deferred_ids)
    assert not ((held_ids | deferred_ids) & set(reg.spawns_by_cid))
    assert set(final["done"]) == {r["configuration_ids"]["2048"] for r in LAUNCH}
    assert final["queued"] == [] and final["active"] == {} and final["refused"] == {} and final["failed"] == {}
    # the per-configuration record at dispatch_paths[cap], on disk, DONE
    for r in LAUNCH:
        rec = json.loads((w.local_root / r["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))
        assert rec["state"] == "done" and rec["status"] == "OK" and rec["configuration_id"] == r["configuration_ids"]["2048"]
        assert rec["out_dir"] == f"{WAVE_ROOT}/{r['out_root']}" and rec["item_subset"] is None
        assert rec["fc_id"] in reg.calls and rec["function"] == reg.calls[rec["fc_id"]].fn_name
    assert not any((w.local_root / r["dispatch_paths"]["2048"]).exists() for r in HELD + DEFERRED_ROWS)
    # one line per transition in the log, and a summary
    # the spawn is two durable steps — the reservation, then the activation with the FunctionCall id
    assert sum(1 for s in w.log if "queued -> launching" in s) == 30 == sum(1 for s in w.log if "launching -> active" in s)
    assert not any("queued -> active" in s for s in w.log)
    assert sum(1 for s in w.log if "active -> done" in s) == 30
    assert any("summary run" in s and "done 30" in s and "held 10" in s for s in w.log)


# --------------------------------------------------------------------------- #
# (j): the GPU kwargs                                                          #
# --------------------------------------------------------------------------- #
def test_kwargs_name_object_sha256_for_lens_rows_and_none_for_base(tmp_path):
    w = Wave(tmp_path)
    assert w.run() == D.EXIT_OK
    run_id = w.state()["run_id"]
    fan_out_keys = {"sha", "limit", "inspect_limit", "samples", "pass_no", "controls", "reduction_mode",
                    "object_digests", "oos_prompt_budget", "run_id", "ladder"}
    n_lens = n_base = 0
    for fc in w.registry.calls.values():
        row = ROW_BY_CID[fc.cfg["configuration_id"]]
        kw = fc.kwargs
        assert set(kw) == fan_out_keys
        assert kw["sha"] == SHA and kw["run_id"] == run_id and kw["pass_no"] == 1 and kw["ladder"] is None
        assert kw["controls"] == [C.CONTROL_IFEVAL] and kw["reduction_mode"] == C.REDUCTION_MODE_STACKED
        assert kw["samples"] == C.IFEVAL_SAMPLES_PER_ITEM and kw["oos_prompt_budget"] is None
        assert kw["limit"] == 0 and kw["inspect_limit"] == 0
        if row["object_sha256"]:
            n_lens += 1
            assert kw["object_digests"] == {row["cell"]: {"artifact_sha256": row["object_sha256"], "intervention_hash": None,
                                                          "expected": None, "lens_route_intervention_hash": None}}
        else:
            n_base += 1
            assert kw["object_digests"] is None
    assert n_base == 17 and n_lens == 13          # 7 none + 10 prompted (no object); 13 LoRA objects
    # every planned configuration: the IFEval tuple, pass 1, the row's view and out root
    for call in w.launcher.plan_calls:
        assert call["modes"] == [C.THINKING_OFF] and call["controls"] == [C.CONTROL_IFEVAL]
        assert call["sampling_mode"] == C.SAMPLING_VENDOR_PRESET and call["samples"] == C.IFEVAL_SAMPLES_PER_ITEM
        assert call["reduction_mode"] == C.REDUCTION_MODE_STACKED and call["pass_no"] == 1
        assert call["off_cap_by_task"] is None and call["item_subset"] is None


# --------------------------------------------------------------------------- #
# (c): a failed call releases its slot                                         #
# --------------------------------------------------------------------------- #
def test_failed_call_releases_slot_and_next_queued_row_starts(tmp_path):
    seen = {}

    def on_tick(w, st, tick):
        if tick == 1:
            assert len(st["active"]) == 8 and len(w.registry.order) == 8
            w.registry.finish(w.registry.calls["fc-000"], error=RuntimeError("container crashed"))
            w.complete_per_tick = 0            # nothing else finishes this tick
        elif tick == 2:
            seen["failed"] = dict(st["failed"])
            seen["active"] = len(st["active"])
            seen["spawned"] = len(w.registry.order)
            w.complete_per_tick = 5

    w = Wave(tmp_path, complete_per_tick=0, on_tick=on_tick)
    rc = w.run()
    assert rc == D.EXIT_FAILED
    first_cid = w.registry.calls["fc-000"].cfg["configuration_id"]
    assert set(seen["failed"]) == {first_cid} and "RuntimeError" in seen["failed"][first_cid]["error"]
    assert seen["active"] == 8 and seen["spawned"] == 9, "the freed slot was refilled by the next queued row"
    assert w.registry.peak == 8
    final = w.state()
    assert set(final["failed"]) == {first_cid} and len(final["done"]) == 29
    assert any(f"{first_cid}: active -> failed" in s for s in w.log)


# --------------------------------------------------------------------------- #
# (d): a refusal REC is filed REFUSED and never re-spawned                     #
# --------------------------------------------------------------------------- #
def test_refusal_rec_filed_refused_and_never_respawned(tmp_path):
    nemo_cid = CID[("nemotron3nano4b-sdfbase-main", "none")]

    def on_tick(w, st, tick):
        for fc in w.registry.unfinished():
            if fc.cfg["configuration_id"] == nemo_cid:
                w.registry.finish(fc, rec={"status": C.REC_NEMO_BACKEND_UNATTESTED, "name": fc.cfg["name"]})

    w = Wave(tmp_path, on_tick=on_tick)
    assert w.run() == D.EXIT_FAILED
    st = w.state()
    assert set(st["refused"]) == {nemo_cid} and st["refused"][nemo_cid]["status"] == C.REC_NEMO_BACKEND_UNATTESTED
    assert w.registry.calls[[i for i in w.registry.order if w.registry.calls[i].cfg["configuration_id"] == nemo_cid][0]] \
        .fn_name == "run_configuration_nemotron"
    assert len(st["done"]) == 29 and w.registry.spawns_by_cid[nemo_cid] == 1
    rec = json.loads((w.local_root / ROW_BY_CID[nemo_cid]["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))
    assert rec["state"] == "refused" and rec["status"] == C.REC_NEMO_BACKEND_UNATTESTED
    # a re-run over the same state file: the refused configuration stays refused, nothing is spawned
    w2 = Wave(tmp_path, registry=w.registry)
    assert w2.run() == D.EXIT_FAILED
    assert len(w2.launcher.launch_calls) == 0 and w.registry.spawns_by_cid[nemo_cid] == 1
    assert set(w2.state()["refused"]) == {nemo_cid}


# --------------------------------------------------------------------------- #
# (e): a restart adopts the active set by fc id; a validated store is skipped  #
# --------------------------------------------------------------------------- #
def test_restart_adopts_active_by_fc_id_and_skips_validated_store(tmp_path):
    w1 = Wave(tmp_path, max_ticks=1)
    assert w1.run() == D.EXIT_FAILED                 # the guard: state persisted with 8 active, 22 queued
    st1 = w1.state()
    assert len(st1["active"]) == 8 and len(st1["queued"]) == 22 and len(w1.registry.order) == 8
    active_fc_ids = {ent["fc_id"] for ent in st1["active"].values()}
    assert active_fc_ids == set(w1.registry.order)
    validated = LAUNCH[-1]["configuration_ids"]["2048"]
    adopted: List[str] = []

    def adopt(fc_id):
        adopted.append(fc_id)
        return w1.registry.calls[fc_id]

    w2 = Wave(tmp_path, registry=w1.registry, adopt=adopt,
              store_complete=lambda cfg: cfg["configuration_id"] == validated)
    assert w2.state()["run_id"] == st1["run_id"]      # the same wave, the same run id
    assert w2.run() == D.EXIT_OK
    assert set(adopted) == active_fc_ids
    for cid in st1["active"]:
        assert w1.registry.spawns_by_cid[cid] == 1, "an adopted configuration is never re-spawned"
    assert validated not in w1.registry.spawns_by_cid and len(w1.registry.order) == 29
    st2 = w2.state()
    assert len(st2["done"]) == 30 and st2["done"][validated]["status"] == "DONE"
    assert any(f"{validated}: queued -> done" in s and "store validated" in s for s in w2.log)
    assert sum(1 for s in w2.log if "adopted FunctionCall" in s) == 8
    # a state file from another wave is refused, never silently re-used
    other = json.loads(w2.state_path.read_text(encoding="utf-8"))
    other["sha"] = "f" * 40
    w2.state_path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="different wave"):
        Wave(tmp_path, registry=w1.registry)


def test_default_store_complete_reads_the_dispatch_record(tmp_path):
    w = Wave(tmp_path)
    assert w.run() == D.EXIT_OK
    rows_by_id = {r["configuration_ids"]["2048"]: r for r in LAUNCH}
    complete = D.default_store_complete(w.local_root, 2048, rows_by_id, ("OK", "DONE"))
    done_cid = LAUNCH[0]["configuration_ids"]["2048"]
    assert complete({"configuration_id": done_cid}) is True
    assert complete({"configuration_id": "no/such/rung2048"}) is False
    # a FAILED record does not validate

    def fail_one_once(w, st, tick):
        if tick == 1:
            w.registry.finish(w.registry.unfinished()[0], error=RuntimeError("container crashed"))

    w2 = Wave(tmp_path / "other", on_tick=fail_one_once)
    assert w2.run() == D.EXIT_FAILED
    failed_cid = next(iter(w2.state()["failed"]))
    assert D.default_store_complete(w2.local_root, 2048, rows_by_id, ("OK", "DONE"))({"configuration_id": failed_cid}) is False
    # and a re-run re-queues the FAILED configuration (retryable) while done ones stay done: the queue
    # is rebuilt in memory at construction and persisted when the loop starts
    w3 = Wave(tmp_path / "other", registry=w2.registry)
    assert set(w3.d.state["queued"]) == {failed_cid} and w3.d.state["retried"][failed_cid]["error"]
    assert len(w3.d.state["done"]) == 29 and failed_cid not in w3.d.state["failed"]


# --------------------------------------------------------------------------- #
# (k): a 4096-rung row's planning inputs                                       #
# --------------------------------------------------------------------------- #
def test_rung_4096_planning_inputs(tmp_path):
    row = next(r for r in LAUNCH if r["slug"] == "gemma4e4b-sdfbase-main" and r["object_sha256"])
    cid = row["configuration_ids"]["4096"]
    reg = FakeRegistry()
    L = make_launcher(reg)
    mpath = tmp_path / "matrix.json"
    mpath.write_text(json.dumps(MATRIX), encoding="utf-8")
    local = tmp_path / "local"

    def sleep(_s):
        for fc in reg.unfinished():
            reg.finish(fc, rec={"status": "OK"})

    preflights = []

    def run_subprocess(argv, env=None, **kw):                 # the admission's preflight per (slug, cap)
        preflights.append(list(argv))
        return types.SimpleNamespace(returncode=0)

    rc = D.main(["--matrix", str(mpath), "--sha", SHA, "--rung", "4096", "--rows", cid, "--wave-root", WAVE_ROOT,
                 "--local-root", str(local), "--app-name", APP], launcher=L, sleep=sleep, clock=FakeClock(),
                adopt=lambda fc_id: reg.calls[fc_id], store_complete=lambda cfg: False, run_subprocess=run_subprocess,
                log=lambda s: None)
    assert rc == D.EXIT_OK and len(reg.order) == 1
    (pf,) = [a for a in preflights if a[-1] == "--pf"]
    assert pf[pf.index(C.FLAG_OFF_CAP_BY_TASK) + 1] == f"{C.IFEVAL_TASK_NAME}=4096" and pf[pf.index("--cells") + 1] == row["cell"]
    (call,) = L.plan_calls
    assert call["cells"] == [row["cell"]] and call["modes"] == [C.THINKING_OFF] and call["controls"] == [C.CONTROL_IFEVAL]
    assert call["model"] == row["slug"] and call["bank_view"] == row["view"] == "main"
    assert call["off_cap_by_task"] == {C.IFEVAL_TASK_NAME: 4096}
    assert call["item_subset"] == f"{WAVE_ROOT}/{row['subset_paths']['4096']}"
    assert call["out_dir"] == f"{WAVE_ROOT}/{row['rung_out_roots']['4096']}"
    assert call["sampling_mode"] == C.SAMPLING_VENDOR_PRESET and call["samples"] == C.IFEVAL_SAMPLES_PER_ITEM
    assert call["reduction_mode"] == C.REDUCTION_MODE_STACKED and call["pass_no"] == 1
    fc = reg.calls[reg.order[0]]
    assert fc.kwargs["object_digests"][row["cell"]]["artifact_sha256"] == row["object_sha256"]
    rec = json.loads((local / row["dispatch_paths"]["4096"]).read_text(encoding="utf-8"))
    assert rec["state"] == "done" and rec["cap"] == 4096 and rec["item_subset"] == call["item_subset"]
    assert rec["geometry_eval"] == "capability:ifeval_4096"
    st = json.loads((local / D.IFEVAL_DISPATCH_STATE).read_text(encoding="utf-8"))
    assert st["cap"] == 4096 and set(st["done"]) == {cid} and len(st["held"]) == 2 + 8
    # --rows naming a held row, a deferred slot, or an unknown one, refuses
    held_cid = HELD[0]["configuration_ids"]["4096"]
    with pytest.raises(ValueError, match="held"):
        D.select_rows(MATRIX, cap=4096, rows=[held_cid])
    deferred_cid = DEFERRED_ROWS[0]["configuration_ids"]["4096"]
    with pytest.raises(ValueError, match="held or deferred"):
        D.select_rows(MATRIX, cap=4096, rows=[deferred_cid])
    with pytest.raises(ValueError, match="unknown"):
        D.select_rows(MATRIX, cap=4096, rows=["nope/none/rung4096"])
    with pytest.raises(ValueError, match="ladder"):
        D.select_rows(MATRIX, cap=16384)


def test_select_rows_refuses_unknown_status():
    bad = json.loads(json.dumps(MATRIX))
    bad["rows"][3]["status"] = "maybe"
    with pytest.raises(ValueError, match="not launch | held | deferred"):
        D.select_rows(bad)


def test_deferred_slots_are_never_launched_and_a_filled_one_must_be_regenerated():
    """The follow-up: a DEFERRED row (an AlphaSteer / Arditi slot with no object sha) never enters the queue —
    it is returned among the held rows with its record path as the reason — and a slot whose sha somebody filled
    by hand refuses by name (the matrix is regenerated; a filled slot is a launch row)."""
    launch, held = D.select_rows(MATRIX)
    assert len(launch) == 30 and len(held) == 10
    assert {r["cell"] for r in held if r["status"] == "deferred"} == {"alphasteer", "arditi"}
    for r in held:
        if r["status"] == "deferred":
            assert r["object_sha256"] is None and r["held_reason"].startswith("DEFERRED:") and r["deferred"]["record"] in r["held_reason"]
    assert not any(r["status"] == "deferred" for r in launch)
    filled = json.loads(json.dumps(MATRIX))
    slot = next(r for r in filled["rows"] if r["status"] == "deferred")
    slot["object_sha256"] = "ab" * 32
    with pytest.raises(ValueError, match="status deferred but an object_sha256 is filled"):
        D.select_rows(filled)
    # the frozen matrix's real deferred rows read the same way through the launcher-side reader
    frozen = json.loads((REPO / "configs/sdfbase/ifeval/matrix.json").read_text(encoding="utf-8"))
    launch_f, held_f = D.select_rows(frozen)
    assert len(launch_f) == 30 and sum(1 for r in held_f if r["status"] == "deferred") == 8
    assert all(r["status"] == "launch" for r in launch_f)


# --------------------------------------------------------------------------- #
# (h) --dry-run, (i) --stop, the STOP sentinel in the loop                     #
# --------------------------------------------------------------------------- #
def test_dry_run_spawns_nothing(tmp_path):
    reg = FakeRegistry()
    L = make_launcher(reg)
    mpath = tmp_path / "matrix.json"
    mpath.write_text(json.dumps(MATRIX), encoding="utf-8")
    out: List[str] = []
    rc = D.main(["--matrix", str(mpath), "--sha", SHA, "--wave-root", WAVE_ROOT, "--local-root", str(tmp_path / "local"),
                 "--app-name", APP, "--dry-run"], launcher=L, log=out.append)
    assert rc == D.EXIT_OK
    assert reg.order == [] and L.launch_calls == []
    assert not (tmp_path / "local" / D.IFEVAL_DISPATCH_STATE).exists()
    text = "\n".join(out)
    for r in LAUNCH:
        assert r["configuration_ids"]["2048"] in text and f"{WAVE_ROOT}/{r['out_root']}" in text
    assert "run_configuration_nemotron" in text and "run_configuration |" in text
    assert sum(1 for s in out if s.startswith("  HELD ")) == 2 + 8 and any("nothing spawned" in s for s in out)
    assert sum(1 for s in out if s.startswith("  HELD ") and "DEFERRED:" in s) == 8


def test_stop_writes_sentinel_and_stops_the_app(tmp_path):
    calls = []

    def run_subprocess(argv, **kw):
        calls.append(list(argv))
        return types.SimpleNamespace(returncode=0)

    local = tmp_path / "local"
    rc = D.main(["--matrix", "unused.json", "--sha", SHA, "--local-root", str(local), "--app-name", APP, "--stop"],
                launcher=None, run_subprocess=run_subprocess, log=lambda s: None)
    assert rc == D.EXIT_OK
    assert (local / D.STOP_SENTINEL).is_file() and APP in (local / D.STOP_SENTINEL).read_text(encoding="utf-8")
    assert calls == [[*D.STOP_APP_ARGV, APP, "--yes"]] and D.STOP_APP_ARGV[:3] == (sys.executable, "-m", "modal")
    # a dispatcher started under the standing sentinel spawns nothing and exits 4
    w = Wave(tmp_path)
    assert w.run() == D.EXIT_STOPPED and w.registry.order == []


def test_running_loop_honours_the_stop_sentinel(tmp_path):
    def on_tick(w, st, tick):
        if tick == 1:
            (w.state_path.parent / D.STOP_SENTINEL).write_text("stop\n", encoding="utf-8")

    w = Wave(tmp_path, complete_per_tick=0, on_tick=on_tick)
    assert w.run() == D.EXIT_STOPPED
    assert len(w.registry.order) == 8 and len(w.launcher.launch_calls) == 8
    st = w.state()
    assert len(st["active"]) == 8 and len(st["queued"]) == 30 - 8
    assert any("STOP" in s for s in w.log)


# --------------------------------------------------------------------------- #
# --preflight: per slug, the launcher's own argv + --pf, the app name in the env #
# --------------------------------------------------------------------------- #
def test_preflight_argv_and_gate(tmp_path):
    reg = FakeRegistry()
    L = make_launcher(reg)
    argv, env = D.preflight_argv(L, sha=SHA, app_name=APP, slug="gemma4e4b-sdfbase-main", cells=["none", "safety_refusal__lora_r4"])
    # this interpreter's `-m modal`, the launcher script by absolute repository path
    assert argv[:5] == [sys.executable, "-m", "modal", "run", str(D.LAUNCHER_SCRIPT)] and argv[-1] == "--pf"
    assert Path(argv[4]).is_absolute() and Path(argv[4]).is_file()
    assert not argv[0].startswith(C.MODAL_APP_NAME_ENV) and env[C.MODAL_APP_NAME_ENV] == APP
    assert argv[argv.index("--cells") + 1] == "none,safety_refusal__lora_r4"
    assert argv[argv.index("--model") + 1] == "gemma4e4b-sdfbase-main"
    assert argv[argv.index("--reduction-mode") + 1] == C.REDUCTION_MODE_STACKED
    assert argv[argv.index("--modes") + 1] == C.THINKING_OFF and argv[argv.index("--controls") + 1] == C.CONTROL_IFEVAL
    assert "--pass1" not in argv and "--pass2" not in argv
    runs = []

    def run_subprocess(argv, env=None, **kw):
        runs.append((argv, env))
        return types.SimpleNamespace(returncode=1 if "nemotron3nano4b-sdfbase-main" in argv else 0)

    mpath = tmp_path / "matrix.json"
    mpath.write_text(json.dumps(MATRIX), encoding="utf-8")
    rc = D.main(["--matrix", str(mpath), "--sha", SHA, "--wave-root", WAVE_ROOT, "--local-root", str(tmp_path / "local"),
                 "--app-name", APP, "--preflight"], launcher=L, run_subprocess=run_subprocess, log=lambda s: None)
    assert rc == D.EXIT_FAILED and reg.order == []
    assert len(runs) == 7 and all(env[C.MODAL_APP_NAME_ENV] == APP for _, env in runs)
    slugs = [a[a.index("--model") + 1] for a, _ in runs]
    # one preflight per slug of THIS (synthetic, frozen-shaped) matrix: the seven instrument-view slugs — the register also
    # carries the six seed lane-view slugs, which this matrix does not express
    assert len(set(slugs)) == 7 and set(slugs) == {r["slug"] for r in MATRIX["rows"]} < set(C.IFEVAL_SLUGS.values())
    assert all(C.FLAG_OFF_CAP_BY_TASK not in a for a, _ in runs)                    # pass 1: the table's 2,048


# --------------------------------------------------------------------------- #
# the remote work outlives the dispatcher — a DETACHED app, persisted           #
# app ids, explicit STOP, clean-up on completion                                #
# --------------------------------------------------------------------------- #
class FakeApp:
    """A double of ``modal.App`` whose ``run`` behaves like the installed client's runner: an EPHEMERAL run
    (``detach=False``, the default) TERMINATES every task still running when the context exits ("Terminates all
    running tasks for ephemeral apps", ``modal/runner.py``); a DETACHED run leaves them running. Each run mints a new
    app id, as Modal does."""

    def __init__(self, registry: FakeRegistry):
        self.registry, self.runs, self.n = registry, [], 0

    def run(self, *, detach: bool = False, **kw):
        self.n += 1
        app_id = f"ap-{self.n}"
        self.runs.append({"detach": detach, "app_id": app_id, **kw})
        registry, runs = self.registry, self.runs

        @contextlib.contextmanager
        def _ctx():
            yield types.SimpleNamespace(app_id=app_id)
            if not detach:
                for fc in registry.unfinished():
                    registry.finish(fc, error=RuntimeError("cancelled: the ephemeral app's client disconnected"))
            runs[-1]["exited"] = True
        return _ctx()


def _matrix_file(tmp_path: Path) -> Path:
    mpath = tmp_path / "matrix.json"
    mpath.write_text(json.dumps(MATRIX), encoding="utf-8")
    return mpath


def test_the_app_context_is_detached_and_the_ephemeral_double_really_cancels_running_calls():
    reg = FakeRegistry()
    L = make_launcher(reg)
    L.app = FakeApp(reg)
    # the double bites: an EPHEMERAL run cancels what is still running at exit ...
    with FakeApp(reg).run() as running:
        fc = L.run_configuration.spawn({"name": "x", "cell": "none", "configuration_id": "c"}, sha=SHA)
        assert running.app_id == "ap-1"
    assert fc.finished
    with pytest.raises(RuntimeError, match="cancelled: the ephemeral app"):
        fc.get(timeout=0)
    # ... and the PRODUCTION context asks for a DETACHED app, under which the same call keeps running
    with D._app_context(L) as running:
        fc2 = L.run_configuration.spawn({"name": "y", "cell": "none", "configuration_id": "d"}, sha=SHA)
        assert D.app_id_of(running) == "ap-1" and L.app.runs[-1]["detach"] is True
    assert not fc2.finished and reg.unfinished() == [fc2]
    assert D.app_id_of(None) is None                                        # a stub launcher's null context
    assert isinstance(D._app_context(make_launcher(reg)), contextlib.nullcontext)
    src = (REPO / "scripts" / "tinycap_ifeval_dispatch.py").read_text(encoding="utf-8")
    code = [l.strip() for l in src.splitlines() if l.strip().startswith(("return ", "with ")) and "app.run(" in l]
    assert code == ['return app.run(detach=True) if app is not None and hasattr(app, "run") else contextlib.nullcontext()']


def test_a_guard_exit_leaves_the_calls_running_and_a_restart_adopts_them_under_a_new_app_then_cleans_up(tmp_path):
    """Shutdown / restart through ``main`` with the cancelling double: the first run exits on the tick guard with eight
    calls active — under the detached app they are STILL RUNNING (an ephemeral app would have cancelled them) and the
    state names the app on every active entry; the second run adopts them by FunctionCall id under a NEW detached app,
    completes the wave, and stops BOTH apps by id (the clean-completion clean-up — nothing active, so nothing is
    cancelled)."""
    reg = FakeRegistry()
    L = make_launcher(reg)
    L.app = FakeApp(reg)
    mpath, local = _matrix_file(tmp_path), tmp_path / "local"
    calls: List[List[str]] = []

    def run_subprocess(argv, **kw):
        calls.append(list(argv))
        return types.SimpleNamespace(returncode=0)

    def stops():
        return [a for a in calls if a[:len(D.STOP_APP_ARGV)] == list(D.STOP_APP_ARGV)]

    base = ["--matrix", str(mpath), "--sha", SHA, "--wave-root", WAVE_ROOT, "--local-root", str(local), "--app-name", APP,
            "--rows", ",".join(CID[(r["slug"], r["cell"])] for r in LAUNCH[:10])]
    rc = D.main(base + ["--max-ticks", "1"], launcher=L, sleep=lambda s: None, clock=FakeClock(),
                adopt=lambda fc_id: reg.calls[fc_id], run_subprocess=run_subprocess, log=lambda s: None)
    assert rc == D.EXIT_FAILED                                              # the guard, not a completion
    assert L.app.runs == [{"detach": True, "app_id": "ap-1", "exited": True}]
    assert len(reg.unfinished()) == 8 and all(not fc.finished for fc in reg.calls.values())   # NOT cancelled
    st = json.loads((local / D.IFEVAL_DISPATCH_STATE).read_text(encoding="utf-8"))
    assert st["app_id"] == "ap-1" and st["app_ids"] == ["ap-1"] and len(st["active"]) == 8 and len(st["queued"]) == 2
    assert {e["app_id"] for e in st["active"].values()} == {"ap-1"} and stops() == []
    rec = json.loads((local / LAUNCH[0]["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))
    assert rec["app_id"] == "ap-1" and rec["state"] == "active"
    # the restart: adopt by id under a NEW app, finish everything, clean up both apps
    adopted: List[str] = []

    def adopt(fc_id):
        adopted.append(fc_id)
        return reg.calls[fc_id]

    def sleep(_s):
        for fc in reg.unfinished()[:4]:
            reg.finish(fc, rec={"status": "OK", "name": fc.cfg["name"], "cell": fc.cfg["cell"]})

    rc = D.main(base, launcher=L, sleep=sleep, clock=FakeClock(), adopt=adopt, run_subprocess=run_subprocess,
                log=lambda s: None)
    assert rc == D.EXIT_OK and sorted(adopted) == sorted(st["active"][c]["fc_id"] for c in st["active"])
    assert [r["app_id"] for r in L.app.runs] == ["ap-1", "ap-2"] and all(r["detach"] for r in L.app.runs)
    st2 = json.loads((local / D.IFEVAL_DISPATCH_STATE).read_text(encoding="utf-8"))
    assert st2["app_id"] == "ap-2" and st2["app_ids"] == ["ap-1", "ap-2"] and len(st2["done"]) == 10
    assert not st2["active"] and not st2["queued"] and reg.spawns_by_cid.most_common(1)[0][1] == 1   # no re-spawn
    assert {e["app_id"] for e in st2["done"].values()} == {"ap-1", "ap-2"}
    assert stops() == [[*D.STOP_APP_ARGV, "ap-1", "--yes"], [*D.STOP_APP_ARGV, "ap-2", "--yes"]]
    assert st2["apps_stopped"] == {"ap-1": 0, "ap-2": 0}


def test_stop_cancels_every_recorded_app_by_id_and_cleanup_refuses_while_calls_run(tmp_path):
    calls: List[List[str]] = []

    def run_subprocess(argv, **kw):
        calls.append(list(argv))
        return types.SimpleNamespace(returncode=0 if "ap-2" not in argv else 3)

    local = tmp_path / "local"
    local.mkdir()
    (local / D.IFEVAL_DISPATCH_STATE).write_text(json.dumps(
        {"schema": D.STATE_SCHEMA, "app_ids": ["ap-1"], "app_id": "ap-2",
         "active": {"c": {"fc_id": "fc-000", "app_id": "ap-2"}, "d": {"fc_id": "fc-001", "app_id": "ap-0"}}}), encoding="utf-8")
    assert D.recorded_app_ids(local / D.IFEVAL_DISPATCH_STATE) == ["ap-1", "ap-2", "ap-0"]
    rc = D.main(["--matrix", "unused.json", "--sha", SHA, "--local-root", str(local), "--app-name", APP, "--stop"],
                launcher=None, run_subprocess=run_subprocess, log=lambda s: None)
    assert rc == D.EXIT_FAILED                                              # one stop failed -> reported
    assert calls == [[*D.STOP_APP_ARGV, i, "--yes"] for i in ("ap-1", "ap-2", "ap-0")]
    assert (local / D.STOP_SENTINEL).is_file()
    # a torn / absent state falls back to the shared name (as earlier waves did)
    assert D.recorded_app_ids(tmp_path / "nowhere.json") == []
    # clean-up never runs over active or queued calls
    w = Wave(tmp_path / "w", complete_per_tick=0, max_ticks=1)
    assert w.run() == D.EXIT_FAILED and len(w.state()["active"]) == 8
    with pytest.raises(RuntimeError, match="stopping an app now would cancel running calls"):
        w.d.cleanup_apps()
    # ... and a wave with no recorded app (a stub launcher) cleans up nothing
    w2 = Wave(tmp_path / "w2")
    assert w2.run() == D.EXIT_OK and w2.d.cleanup_apps() == {} and w2.state()["app_ids"] == []


# --------------------------------------------------------------------------- #
# the seed sibling document plans NATIVELY                                     #
# --------------------------------------------------------------------------- #
SEED_MATRIX = json.loads((REPO / C.IFEVAL_MATRIX_QWEN_SEEDREP_REL).read_text(encoding="utf-8"))
SEED_WAVE_ROOT = f"{C.VOL_TINYCAP_ROOT}/ifeval_seedrep/{SHA7}"


def test_the_seed_sibling_document_plans_every_row_natively_and_the_frozen_rows_pass_no_serving_path(tmp_path):
    """The REAL sibling document (30 rows, `serving_path: lora_native` stamped on every row) through the dispatcher's doors:
    `select_rows` admits all 30 (0 held); `plan_row` passes `serving_path="lora_native"` to the launcher's planner and the
    returned configuration carries the launcher's native marker; the preflight argv hands the rows' stamp to the argv
    composer; the dispatch record carries it. A frozen-matrix row (no stamp) passes NO serving-path keyword at all — its
    plan call is unchanged from before the stamp existed — and a row with an unknown stamp / a mixed set refuses by name."""
    launch, held = D.select_rows(SEED_MATRIX)
    assert len(launch) == 30 and held == []
    assert [r["configuration_ids"]["2048"] for r in launch][:2] == ["qwen35_4b_s1/none/rung2048",
                                                                     "qwen35_4b_s1/constv2-mix-full--psr_ce_all/rung2048"]
    assert all(D.row_serving_path(r) == "lora_native" for r in launch) and D.rows_serving_path(launch) == "lora_native"
    L = make_launcher(FakeRegistry())
    anchor = launch[0]
    cfg = D.plan_row(L, anchor, 2048, wave_root=SEED_WAVE_ROOT)
    assert L.plan_calls[-1]["serving_path"] == "lora_native" and L.plan_calls[-1]["bank_view"] == "s1"
    # the native engine admits only the launcher's DEFAULT reduction mode (its dry run refuses `stacked` under lora_native):
    # the plan, the GPU kwargs and the preflight argv of a native row all carry the default; a lens row keeps `stacked`
    assert L.plan_calls[-1]["reduction_mode"] == C.REDUCTION_MODE_LAUNCHER_DEFAULT != C.REDUCTION_MODE_STACKED
    assert D.kwargs_for(anchor, sha=SHA, run_id="r")["reduction_mode"] == C.REDUCTION_MODE_LAUNCHER_DEFAULT
    assert D.row_reduction_mode(anchor) == C.REDUCTION_MODE_LAUNCHER_DEFAULT and D.row_reduction_mode(LAUNCH[0]) == C.REDUCTION_MODE_STACKED
    assert D.kwargs_for(LAUNCH[0], sha=SHA, run_id="r")["reduction_mode"] == C.REDUCTION_MODE_STACKED
    assert cfg["serving_path"] == "lora_native" and cfg["model"] == "qwen35_4b-sdfbase-s1" and cfg["cell"] == "none"
    # the explicit out root is the ONE path constructor's under the wave root — left as named (never re-rooted)
    assert cfg["out_dir"] == f"{SEED_WAVE_ROOT}/controls/qwen35_4b_s1/none/vendor_preset"
    arditi = next(r for r in launch if r["cell"] == "arditi_allpos" and r["view"] == "s2")
    cfg_a = D.plan_row(L, arditi, 2048, wave_root=SEED_WAVE_ROOT)
    assert cfg_a["serving_path"] == "lora_native" and cfg_a["bank_view"] == "s2" and cfg_a["steered"]
    assert D.object_digests_for(arditi)["arditi_allpos"]["artifact_sha256"] == arditi["object_sha256"]
    # the composed row on a paired view: its own slug, its own view; the anchor is another view's `none` (the reader's concern)
    composed = next(r for r in launch if r["view"] == "s1p")
    assert composed["cell"] == "constv2-mix-full__psr_ce_all_minus_ctl" and not any(r["view"] == "s1p" and r["cell"] == "none" for r in launch)
    # the preflight hands the rows' stamp to the argv composer, per slug
    argv, _env = D.preflight_argv(L, sha=SHA, app_name="ifeval-seedrep-x", slug=anchor["slug"], cells=["none"], serving_path="lora_native")
    assert argv[argv.index("--serving-path") + 1] == "lora_native" and argv[-1] == D.PREFLIGHT_FLAG
    assert argv[argv.index("--reduction-mode") + 1] == C.REDUCTION_MODE_LAUNCHER_DEFAULT
    argv_lens, _ = D.preflight_argv(L, sha=SHA, app_name=APP, slug=LAUNCH[0]["slug"], cells=["none"])
    assert argv_lens[argv_lens.index("--reduction-mode") + 1] == C.REDUCTION_MODE_STACKED and "--serving-path" not in argv_lens
    runs = []
    adm = D.run_preflights(L, launch[:2], sha=SHA, app_name="ifeval-seedrep-x",
                           run_subprocess=lambda a, env=None, **k: (runs.append(list(a)), types.SimpleNamespace(returncode=0))[1],
                           log=lambda s: None)
    assert adm["qwen35_4b-sdfbase-s1"]["serving_path"] == "lora_native" and "--serving-path" in runs[0]
    assert L.launch_argv_calls[-1]["serving_path"] == "lora_native"
    # the dispatch record carries the stamp
    w = Wave(tmp_path, max_ticks=0)
    d = D.Dispatcher(L, SEED_MATRIX, sha=SHA, app_name="ifeval-seedrep-x", wave_root=SEED_WAVE_ROOT,
                     local_root=tmp_path / "seed", state_path=tmp_path / "seed" / D.IFEVAL_DISPATCH_STATE, sleep=lambda s: None,
                     clock=FakeClock(), adopt=lambda fc_id: None, store_complete=lambda cfg: False, max_ticks=0, log=lambda s: None)
    cid = anchor["configuration_ids"]["2048"]
    d._write_record(cid, "queued", {})
    rec = json.loads((tmp_path / "seed" / anchor["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))
    # from `queued` the record names what the ROW REQUESTED; the attestation is null until the store is done
    assert rec["requested_serving_path"] == "lora_native" and rec["out_dir"] == cfg["out_dir"] and rec["view"] == "s1"
    assert rec["serving_path_attested"] is None and rec["serving_path_attested_source"] is None and "serving_path" not in rec
    # a FROZEN-matrix row (no stamp): no serving-path keyword reaches the planner, the plan carries no marker, the record no key
    frozen_row = LAUNCH[0]
    cfg_f = D.plan_row(L, frozen_row, 2048, wave_root=WAVE_ROOT)
    assert "serving_path" not in L.plan_calls[-1] and "serving_path" not in cfg_f and D.rows_serving_path(LAUNCH) is None
    w0 = Wave(tmp_path / "frozen", max_ticks=0)
    w0.d._write_record(CID[(frozen_row["slug"], frozen_row["cell"])], "queued", {})
    rec_f = json.loads((w0.local_root / frozen_row["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))
    assert "requested_serving_path" not in rec_f and "serving_path_attested" not in rec_f and "serving_path" not in rec_f
    # an unknown stamp refuses by name; a mixed set refuses by name
    with pytest.raises(ValueError, match="serving_path 'vllm' is not"):
        D.row_serving_path({**anchor, "serving_path": "vllm"})
    with pytest.raises(ValueError, match="more than one serving path"):
        D.rows_serving_path([anchor, frozen_row])
    # the planner not honouring the stamp is a refusal, never a silent lens plan
    L2 = make_launcher(FakeRegistry())
    real_plan = L2.plan_configurations
    L2.plan_configurations = lambda *a, **k: [{**c, "serving_path": "lens"} if "serving_path" in c else c
                                              for c in real_plan(*a, **{kk: vv for kk, vv in k.items() if kk != "serving_path"})]
    with pytest.raises(ValueError, match="did not honour the serving path"):
        D.plan_row(L2, anchor, 2048, wave_root=SEED_WAVE_ROOT)


def _manifest(engine="lora_native", **over):
    """A served store's manifest as the native / lens writers stamp it: the route block's unhashed `serving_engine` and the
    hashed `tinycap.served.serving_engine` (the two spellings contracts.read_store_serving_engine reads first)."""
    doc = {"cell_id": "none", "engine": "vllm", "route": {"serving_engine": engine}, "tinycap": {"served": {"serving_engine": engine}}}
    doc.update(over)
    return doc


def _finish_ok(reg) -> None:
    for fc in reg.unfinished():
        reg.finish(fc, rec={"status": "OK", "name": fc.cfg["name"], "cell": fc.cfg["cell"]})


def test_the_done_record_attests_the_serving_path_from_the_completed_stores_manifest_and_a_mismatch_fails_the_row(tmp_path):
    """`serving_path_attested` is populated ONLY from the completed store's manifest.json (the
    route block's `serving_engine`, else the hashed `tinycap.served.serving_engine`, else the top-level `serving_path`; every
    manifest under the root must agree); a store attesting another engine than the row requested is filed FAILED with
    REC SERVING_PATH_MISMATCH and the wave exits non-zero; an unreadable / torn store fails the same way; a row that requested
    nothing attests nothing and is filed as before."""
    # the store reader: the contracts' ONE serving-engine reader per manifest (agreement across spellings and with the identity is
    # its rule); here: cross-manifest agreement, and a manifest stamping NO engine refuses (never lens by era for this wave's stores)
    attested, source = D.attest_store_serving_path([("a/manifest.json", _manifest()), ("b/manifest.json", _manifest())])
    assert attested == "lora_native" and "route.serving_engine" in source and source.count("=") == 2
    assert D.attest_store_serving_path([("a/manifest.json", {"tinycap": {"served": {"serving_engine": "lens"}}})])[0] == "lens"
    with pytest.raises(ValueError, match="stating no serving_engine|stamps no serving engine"):
        D.attest_store_serving_path([("a/manifest.json", {"cell_id": "none", "engine": "vllm", "route": {}})])
    with pytest.raises(ValueError, match="stamps no serving engine"):
        D.attest_store_serving_path([("a/manifest.json", {"cell_id": "none"})])           # not even a served store: nothing attested
    with pytest.raises(ValueError, match="two ways"):
        D.attest_store_serving_path([("a/manifest.json", {"route": {"serving_engine": "lens"}, "tinycap": {"served": {"serving_engine": "lora_native"}}})])
    with pytest.raises(ValueError, match="not one of"):
        D.attest_store_serving_path([("a/manifest.json", {"route": {"serving_engine": "vllm"}})])
    with pytest.raises(ValueError, match="more than one serving path"):
        D.attest_store_serving_path([("a/manifest.json", _manifest()), ("b/manifest.json", _manifest("lens"))])
    with pytest.raises(ValueError, match="no cells"):
        D.attest_store_serving_path([])
    # the wave: the seed matrix's canary alone, a stub store reader per case
    launch, _held = D.select_rows(SEED_MATRIX)
    anchor = launch[0]
    cid = anchor["configuration_ids"]["2048"]
    seen = []

    def wave(store_manifests):
        reg = FakeRegistry()
        L = make_launcher(reg)
        # one tick spawns; the registry finishes the call with an OK REC on the next (the sleep hook)
        d = D.Dispatcher(L, SEED_MATRIX, sha=SHA, rows=[cid], app_name="ifeval-seedrep-x", wave_root=SEED_WAVE_ROOT,
                         local_root=tmp_path / "w", state_path=tmp_path / "w" / D.IFEVAL_DISPATCH_STATE, sleep=lambda s: _finish_ok(reg),
                         clock=FakeClock(), adopt=lambda fc_id: reg.calls[fc_id], store_complete=lambda cfg: False, max_ticks=6,
                         log=lambda s: None, store_manifests=store_manifests)
        rc = d.run()
        rec = json.loads((tmp_path / "w" / anchor["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))
        return rc, d, rec

    # (a) the store attests what was requested: done, the record carries the attestation and its source
    rc, d, rec = wave(lambda out_dir: (seen.append(out_dir), [(f"{out_dir}/cells/deadbeef/manifest.json", _manifest())])[1])
    assert rc == D.EXIT_OK and cid in d.state["done"] and seen == [f"{SEED_WAVE_ROOT}/controls/qwen35_4b_s1/none/vendor_preset"]
    assert rec["state"] == "done" and rec["requested_serving_path"] == "lora_native"
    assert rec["serving_path_attested"] == "lora_native" and "cells/deadbeef/manifest.json" in rec["serving_path_attested_source"]
    assert "route.serving_engine" in rec["serving_path_attested_source"]
    assert d.state["done"][cid]["serving_path_attested"] == "lora_native"
    # (b) the store attests the OTHER engine: failed by name, non-zero exit, the record carries both readings
    shutil.rmtree(tmp_path / "w")
    rc, d, rec = wave(lambda out_dir: [(f"{out_dir}/cells/deadbeef/manifest.json", _manifest("lens"))])
    assert rc == D.EXIT_FAILED and cid in d.state["failed"] and d.state["failed"][cid]["rec"]["status"] == D.REC_SERVING_PATH_MISMATCH
    assert rec["state"] == "failed" and rec["requested_serving_path"] == "lora_native" and rec["serving_path_attested"] == "lens"
    assert "requested 'lora_native'" in rec["error"] and "attests 'lens'" in rec["error"]
    # (c) an unreadable store (the reader raises) fails the row the same way, attesting nothing
    shutil.rmtree(tmp_path / "w")
    def broken(out_dir):
        raise ValueError("modal volume ls failed (rc 1)")
    rc, d, rec = wave(broken)
    assert rc == D.EXIT_FAILED and rec["state"] == "failed" and rec["serving_path_attested"] is None
    assert "attests nothing readable" in rec["error"] and "modal volume ls failed" in rec["error"]
    # (d) a frozen-matrix row (no request) never consults the store reader
    calls = []
    reg = FakeRegistry(); L = make_launcher(reg)
    fcid = LAUNCH[0]["configuration_ids"]["2048"]
    d0 = D.Dispatcher(L, MATRIX, sha=SHA, rows=[fcid], app_name=APP, wave_root=WAVE_ROOT, local_root=tmp_path / "f",
                      state_path=tmp_path / "f" / D.IFEVAL_DISPATCH_STATE, sleep=lambda s: _finish_ok(reg),
                      clock=FakeClock(), adopt=lambda fc_id: reg.calls[fc_id], store_complete=lambda cfg: False, max_ticks=6,
                      log=lambda s: None, store_manifests=lambda out_dir: (calls.append(out_dir), [])[1])
    assert d0.run() == D.EXIT_OK and fcid in d0.state["done"] and calls == []
    rec0 = json.loads((tmp_path / "f" / LAUNCH[0]["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))
    assert "requested_serving_path" not in rec0 and "serving_path_attested" not in rec0
    # the production reader reads cells/*/manifest.json IN-PROCESS through the volume SDK (listdir + read_file, no subprocess) at the
    # volume-relative path of the container out_dir; a path outside the mount refuses
    class FakeEntry:
        def __init__(self, path, kind): self.path, self.type = path, types.SimpleNamespace(name=kind)

    class FakeVolume:
        calls: list = []
        def listdir(self, path):
            self.calls.append(("listdir", path))
            return [FakeEntry(f"{path}/c1", "DIRECTORY"), FakeEntry(f"{path}/notes.txt", "FILE")]
        def read_file(self, path):
            self.calls.append(("read_file", path))
            yield json.dumps(_manifest()).encode("utf-8")
    vol = FakeVolume()
    got = D.default_store_manifests("/vol/tinycap/ifeval_seedrep/x/controls/a", volume=vol)
    assert [m[0] for m in got] == ["/tinycap/ifeval_seedrep/x/controls/a/cells/c1/manifest.json"] and got[0][1]["route"]["serving_engine"] == "lora_native"
    assert vol.calls == [("listdir", "tinycap/ifeval_seedrep/x/controls/a/cells"), ("read_file", "tinycap/ifeval_seedrep/x/controls/a/cells/c1/manifest.json")]
    assert D.volume_relative("/vol/tinycap/x") == "tinycap/x"
    with pytest.raises(ValueError, match="not under the volume mount"):
        D.volume_relative("/tmp/x")

