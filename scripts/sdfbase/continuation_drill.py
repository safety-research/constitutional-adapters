#!/usr/bin/env python
"""Continuation drill, with a machine-readable verdict.

Modal's 24 h function ceiling cannot be raised, and the gemma full fits
project past it, so continuation — a killed/timed-out fit container
resumed from its volume checkpoint by ONE successor call — is the design,
not a recovery. This drill proves the whole path on a small cell BEFORE any
production spawn, against the SAME deployed code (the wave-A commit under
a throwaway app name with a short fit timeout):

  (a) operator: the wave-A commit deployed under the drill app name with
      the real function timeout shortened to minutes —
          SDFBASE_FIT_TIMEOUT=600 modal deploy --name <drill-app> \\
              scripts/modal/sdfbase_fits.py
      (sdfbase_fits.FIT_TIMEOUT_S reads that env at import; production
      leaves it at 86400). The 1k pool + shared held-out pool must be
      staged for the model. The drill records the deployed function id.
  (b) spawn a 1k cell with a small --checkpoint-every and wait until
      ckpt/<cell>/ckpt.pt appears on the volume with a post-spawn mtime
      (the committer thread's vol.commit landed).
  (c) let that call hit the REAL timeout (state `timeout`, wall recorded);
      separately, a second item is killed by FunctionCall.cancel() after
      its checkpoint lands (state recorded).
  (d) for each ended item the PRODUCTION successor machinery
      (scripts/modal/fits_watch.py: can_respawn / same_app_version /
      spawn_successor / successor_item — imported, not copied) spawns
      EXACTLY ONE successor of the same item on the same app version;
      the per-item cap is then re-checked and must refuse a second.
  (e) each successor's log/result lines must prove continuity: a
      "[fitckpt] ... RESUMING from checkpoint: epoch E, micro-batch M/N in
      epoch, optimizer step S" line with S > 0 (no restart-from-zero), and
      the successor must return `done` with a COMPLETE marker.
  (f) three-way artifact comparison — A uninterrupted, A' a second
      uninterrupted same-seed twin (the nondeterminism floor), A'' the
      interrupted+resumed run — on the parameters (L2 over every float
      tensor leaf):
          pass iff dist(A'', A) <= max(2 * dist(A', A), 1e-9)
      and on the held-out curve's BEST value (the statistic the published
      object is the restore of):
          pass iff |min(A'') - min(A)| <= max(2 * |min(A') - min(A)|, 1e-3)
      with bitwise equality reported when it holds. The max-abs distance
      over the epochs both curves cover is REPORTED as a diagnostic, not
      gated (on a bf16 GPU trajectory the post-minimum tail
      diverges chaotically — the drill's two twins differed by 7.1e-3 at
      the last epoch and a resumed run by 3.8e-2 while every minimum sat
      within 1.3e-3 — so a two-twin max-abs floor fails healthy resumes).
      Twins A / A' are ordinary uninterrupted spawns under their own
      out_roots, launched by the driver alongside the drill items.
  (g) the verdict JSON (every step's evidence, PASS/FAIL per step, overall
      pass iff all steps pass) is written to --out.

  The uninterrupted twins CANNOT run on the drill app: every item is the
  SAME fit, so a function timeout short enough to interrupt the timeout
  item is short enough to interrupt the twins too (found live 2026-09-03:
  twins, timeout item and both successors all ended `timeout` at 600 s,
  and the drill could never have passed). Twins therefore spawn on a
  SECOND throwaway deploy of the SAME commit carrying the production
  timeout (--twin-app; the default, the drill app itself, only works for a
  fit that finishes under the drill timeout — i.e. never when step (c) can
  pass):
          modal deploy --name <drill-app>-long scripts/modal/sdfbase_fits.py
  Successors DO run on the drill app (the production version pin demands
  the row's own function), so the drill timeout must also leave room for a
  successor's remaining work: pick --checkpoint-every so the FIRST
  checkpoint lands late enough that model load + the resumed remainder
  fits inside the timeout (the cancel item is cancelled right after its
  first checkpoint, so its successor redoes everything after it).

Architecture: pure functions + an injectable client (DrillClient) so the
orchestration is CPU-testable with a scripted fake; RealModalClient wraps
modal and is the only place modal is imported. Every item tuple is the
7-tuple fits-spawn sends and fits_watch.successor_item rebuilds —
(model_key, si, arm, config_sha, overwrite, out_root, checkpoint_every) —
pinned at run time against successor_item itself.

Every drill is ONE RUN: its four items live under
``<out-root>/<run-id>/<name>`` on the volume and its comparison artifacts
under ``<compare-root>/<run-id>/<name>`` locally, so a re-run can never
resume a previous drill's checkpoints or compare against a previous drill's
files; a comparison dir that already exists is REFUSED before anything is
spawned. Artifacts are materialized from the CURRENT COMPLETE marker only —
every file the marker lists is fetched and its sha256 verified against the
marker's, a missing or mismatching file fails the compare — and the
three-way PASS requires the interrupted+resumed cell to have published the
SAME file set as the uninterrupted twin (``complete_structure.match``).

Usage (on the grading host). Run with the REPO venv (``.venv/bin/python``):
the comparison loads the object blobs with torch, so an interpreter without
torch (the modal venv) fails after the GPU steps — main() refuses at start
instead, before anything is spawned:
  .venv/bin/python scripts/sdfbase/continuation_drill.py \\
      --app sdfbase-fits-drill --twin-app sdfbase-fits-drill-long \\
      --function fit_h200_std \\
      --model gemma4e4b_base --si constv2-mix-1k --arm psr_ce_all \\
      --config-sha <sha16> --out-root out/drill --checkpoint-every 5 \\
      --function-timeout 600 --poll 30 --max-wait 5400 \\
      [--run-id <token>]   # default: the UTC start time, e.g. 20260902T181500Z
      --out $STAGE/sdfbase/drill_verdict.json
Re-running ONLY the comparison on a run's materialized artifacts (no Modal):
  .venv/bin/python scripts/sdfbase/continuation_drill.py \\
      --compare-only <compare-root>/<run-id> --arm lora_r4_ce --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

HERE = Path(__file__).resolve().parent
MODAL_SCRIPTS_DIR = HERE.parent / "modal"

#: The four drill items, by role. `timeout` / `cancel` are interrupted;
#: `twin_a` / `twin_a2` are the uninterrupted same-seed twins.
INTERRUPTED = ("timeout", "cancel")
TWINS = ("twin_a", "twin_a2")
ITEM_NAMES = INTERRUPTED + TWINS

#: Call states that END a call without a result — the continuation causes
#: (mirrors fits_watch.RESPAWN_CAUSES; asserted equal at run time — this
#: module stays modal-free at import, so it cannot import the watcher's).
ENDED_STATES = ("timeout", "failed")

#: The nondeterminism-floor rule of §3.2.4f.
COMPARE_RULE = "dist(A'', A) <= max(2 * dist(A', A), 1e-9)"
COMPARE_FLOOR = 1e-9
#: The curve leg gates on the best held-out value — the statistic the
#: published object is the restore of — against the two-twin floor with an
#: absolute tolerance of 1e-3 nats (15x tighter than the smallest
#: pre-registered analysis margin, P1's TOST +-0.015), never on the max-abs
#: over epochs (a chaotic-tail readout; reported as a diagnostic).
CURVE_RULE = ("|min(A'') - min(A)| <= max(2 * |min(A') - min(A)|, 1e-3); "
              "max-abs over the common epochs is a diagnostic")
CURVE_MIN_TOL = 1e-3


def _fits_watch():
    """The production successor machinery, imported (never copied) from
    scripts/modal/fits_watch.py. Lazy so the pure helpers of this module
    (resume_proof, three_way_compare) need neither modal nor that path."""
    if str(MODAL_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(MODAL_SCRIPTS_DIR))
    import fits_watch  # noqa: WPS433 — deliberate late import
    return fits_watch


# --------------------------------------------------------------------------- #
# Client protocol + the real modal client                                       #
# --------------------------------------------------------------------------- #


class DrillClient(Protocol):
    """What run_drill needs from Modal. Volume paths are RELATIVE to the
    volume root (the form Volume.listdir / read_file take), e.g.
    `out/drill/timeout/ckpt/<cell>/ckpt.pt`."""

    def spawn(self, item: tuple) -> str: ...
    def state(self, call_id: str) -> str: ...      # running|done|timeout|failed|unknown
    def cancel(self, call_id: str) -> None: ...
    def result(self, call_id: str) -> dict | None: ...
    def logs(self, call_id: str) -> str: ...
    def function_id(self) -> str: ...
    def deployment(self) -> dict | None: ...   # fits_watch.deployment_pin
    def ckpt_mtime(self, vol_path: str) -> float | None: ...
    def read_json(self, vol_path: str) -> dict | None: ...
    def read_bytes(self, vol_path: str) -> bytes | None: ...
    def sleep(self, s: float) -> None: ...


class RealModalClient:
    """DrillClient over modal 1.5.4. `state` reuses fits_watch.classify_call
    so the drill observes calls exactly the way the production watcher
    does. `cancel` terminates the container (a cancelled input whose
    container kept running would race its own successor on the checkpoint
    path). `logs` falls back to the result's `fitckpt_lines` when the logs
    API is unavailable — _fit_impl prints and returns those lines."""

    def __init__(self, app_name: str, fname: str,
                 volume_name: str = "twosi-fits"):
        import modal  # the ONE modal import in this module

        self._modal = modal
        self.app_name, self.fname, self.volume_name = app_name, fname, volume_name
        self._fn = modal.Function.from_name(app_name, fname)
        self._fn.hydrate()
        self._vol = modal.Volume.from_name(volume_name)

    def function_id(self) -> str:
        return self._fn.object_id

    def deployment(self) -> dict | None:
        """The app's live production deployment (the pin that actually
        moves on redeploy); None when it cannot be read, which the
        version pin treats as a refusal."""
        try:
            return _fits_watch().deployment_pin(self.app_name)
        except Exception:  # noqa: BLE001 — unknown deployment = refusal
            return None

    def spawn(self, item: tuple) -> str:
        return self._fn.spawn(item).object_id

    def state(self, call_id: str) -> str:
        return _fits_watch().classify_call({"call_id": call_id})

    def cancel(self, call_id: str) -> None:
        self._modal.FunctionCall.from_id(call_id).cancel(
            terminate_containers=True)

    def result(self, call_id: str) -> dict | None:
        try:
            r = self._modal.FunctionCall.from_id(call_id).get(timeout=0)
        except Exception:  # noqa: BLE001 — not finished / raised / expired
            return None
        return r if isinstance(r, dict) else {"result": r}

    def logs(self, call_id: str) -> str:
        try:
            fc = self._modal.FunctionCall.from_id(call_id)
            entries = list(fc.logs.tail(entries=2000))
            return "\n".join(str(getattr(e, "message", e)) for e in entries)
        except Exception as e:  # noqa: BLE001 — logs API unavailable
            return f"[drill] logs unavailable: {type(e).__name__}: {e}"

    def ckpt_mtime(self, vol_path: str) -> float | None:
        name = vol_path.rstrip("/").rsplit("/", 1)[-1]
        try:
            entries = list(self._vol.listdir(vol_path))
        except Exception:  # noqa: BLE001 — absent
            return None
        for e in entries:
            if str(getattr(e, "path", "")).rstrip("/").endswith(name):
                m = getattr(e, "mtime", None)
                return float(m) if m is not None else None
        return None

    def read_bytes(self, vol_path: str) -> bytes | None:
        try:
            return b"".join(self._vol.read_file(vol_path))
        except Exception:  # noqa: BLE001 — absent
            return None

    def read_json(self, vol_path: str) -> dict | None:
        b = self.read_bytes(vol_path)
        if not b:
            return None
        try:
            return json.loads(b)
        except ValueError:
            return None

    def sleep(self, s: float) -> None:
        time.sleep(s)


# --------------------------------------------------------------------------- #
# Item plumbing                                                                 #
# --------------------------------------------------------------------------- #


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def default_run_id(now: float | None = None) -> str:
    """The UTC start time as a path token (``20260902T181500Z``)."""
    t = datetime.now(timezone.utc) if now is None else \
        datetime.fromtimestamp(now, tz=timezone.utc)
    return t.strftime("%Y%m%dT%H%M%SZ")


def build_items(*, model: str, si: str, arm: str, config_sha: str,
                out_root: str, checkpoint_every: int, compare_root,
                run_id: str, overwrite: bool = False) -> dict:
    """The four drill items of ONE run: one 7-tuple per role under its OWN
    out_root (`<out_root>/<run_id>/<name>` — the four fits share a cell
    id, so the roots keep their checkpoints, cell dirs and reports apart,
    and the run id keeps this drill apart from every previous one: a
    re-run under a previous root would find COMPLETE markers and resume
    the previous drill's checkpoints), plus the local dir
    `<compare_root>/<run_id>/<name>` where that item's artifacts land for
    the comparison."""
    if not run_id or not _RUN_ID_RE.match(run_id):
        raise ValueError(f"run_id {run_id!r} must be a plain path token "
                         "([A-Za-z0-9][A-Za-z0-9_.-]*)")
    items = {}
    for name in ITEM_NAMES:
        items[name] = {
            "item": (model, si, arm, config_sha, bool(overwrite),
                     f"{out_root.rstrip('/')}/{run_id}/{name}",
                     int(checkpoint_every)),
            "dir": str(Path(compare_root).expanduser() / run_id / name),
        }
    return items


def cell_id_of(item: tuple) -> str:
    return f"{item[1]}__{item[2]}"


def ckpt_path(item: tuple) -> str:
    return f"{item[5]}/ckpt/{cell_id_of(item)}/ckpt.pt"


def cell_dir_path(item: tuple) -> str:
    return f"{item[5]}/cells_all/{cell_id_of(item)}"


def complete_path(item: tuple) -> str:
    return f"{cell_dir_path(item)}/COMPLETE"


def report_path(item: tuple) -> str:
    return f"{item[5]}/reports/report_{cell_id_of(item)}_all.json"


def manifest_cell(item: tuple, *, app: str, function: str,
                  function_id: str | None, call_id: str,
                  spawn_time: str, deployment: dict | None = None) -> dict:
    """A manifest ROW for one item — the exact shape fits-spawn writes,
    spawn pin included (app / function / function_id / config_sha /
    overwrite / checkpoint_every ride on the ROW, fits_watch.row_pin; the
    deployment pin app_id / app_version / deploy_commit when the client
    could read it), so the production successor functions run on the same
    input they see in a wave."""
    key, si, arm, sha, overwrite, root, ck = item
    dep = deployment or {}
    return {"model": key, "si": si, "arm": arm, "positions": "all",
            "leg": "all", "cell_id": cell_id_of(item), "out_root": root,
            "app": app, "function": function, "function_id": function_id,
            "config_sha": sha, "overwrite": bool(overwrite),
            "checkpoint_every": int(ck),
            "app_id": dep.get("app_id"), "app_version": dep.get("app_version"),
            "deploy_commit": dep.get("deploy_commit"),
            "call_id": call_id, "spawn_time": spawn_time,
            "status": "spawned (drill)"}


def _client_deployment(client) -> dict | None:
    """The client's live deployment pin, or None for a client that cannot
    read one (an unknown deployment is a version-pin refusal)."""
    fn = getattr(client, "deployment", None)
    if fn is None:
        return None
    try:
        dep = fn()
    except Exception:  # noqa: BLE001
        return None
    return dep if isinstance(dep, dict) else None


def check_item_shape(item: tuple) -> None:
    """Pin: the tuple must round-trip through fits_watch.successor_item
    (the production rebuild of fits-spawn's item), field for field."""
    if not isinstance(item, tuple) or len(item) != 7:
        raise ValueError(f"drill item must be the 7-tuple fits-spawn sends, "
                         f"got {item!r}")
    cell = manifest_cell(item, app="x", function="y", function_id=None,
                         call_id="c", spawn_time="t")
    rebuilt = _fits_watch().successor_item(cell)
    if rebuilt != item:
        raise ValueError(f"item {item!r} does not round-trip through "
                         f"fits_watch.successor_item ({rebuilt!r})")


class _ClientFn:
    """The `fn` fits_watch.spawn_successor expects (`.object_id`,
    `.spawn(item) -> call with .object_id`), over a DrillClient."""

    def __init__(self, client: DrillClient):
        self._client = client
        self.object_id = client.function_id()

    def spawn(self, item: tuple):
        return SimpleNamespace(object_id=self._client.spawn(item))


# --------------------------------------------------------------------------- #
# Resume proof (step e)                                                         #
# --------------------------------------------------------------------------- #

_POS = (r"epoch (?P<epoch>\d+), (?:micro-)?batch (?P<done>\d+)/(?P<n>\d+) "
        r"in epoch, optimizer steps? (?:so far )?(?P<step>\d+)")
RESUME_RE = re.compile(
    r"\[fitckpt\] (?P<trainer>\S+) RESUMING from checkpoint: " + _POS)
WRITTEN_RE = re.compile(
    r"\[fitckpt\] (?P<trainer>\S+) checkpoint written: " + _POS)
LOADED_RE = re.compile(r"\[fitckpt\] loaded checkpoint (?P<path>\S+)")
#: fitckpt.load_checkpoint's refusal vocabulary (a refused slot falls
#: through to the previous one or to a fresh start).
REFUSAL_MARKERS = ("IGNORING it", "starting this fit fresh",
                   "starting fresh", "carries no checkpoint_key",
                   "is unreadable")


def _pos(m: re.Match) -> dict:
    return {"trainer": m.group("trainer"), "epoch": int(m.group("epoch")),
            "batches_done": int(m.group("done")),
            "n_batches": int(m.group("n")), "step": int(m.group("step"))}


def resume_proof(text_lines) -> dict:
    """Parse a successor's `[fitckpt]` lines. `resumed` is True only when a
    RESUMING line is present AND its optimizer step is > 0 (a resume at
    step 0 is indistinguishable from a restart). Also reports the loaded
    checkpoint path, every `checkpoint written` position, and any loader
    refusal lines (a refusal WITH a later RESUMING line means the previous
    slot was used — still a resume; a refusal without one is a fresh
    start)."""
    if isinstance(text_lines, str):
        lines = text_lines.splitlines()
    else:
        lines = [str(ln) for ln in (text_lines or [])]
    out: dict = {"resumed": False, "epoch": None, "batches_done": None,
                 "n_batches": None, "step": None, "trainer": None,
                 "resume_line": None, "loaded_checkpoint": None,
                 "checkpoint_written": [], "refusals": [],
                 "restart_from_zero": False, "reason": None}
    for ln in lines:
        m = RESUME_RE.search(ln)
        if m and out["resume_line"] is None:
            out.update(_pos(m))
            out["resume_line"] = ln.strip()
            continue
        m = WRITTEN_RE.search(ln)
        if m:
            out["checkpoint_written"].append(_pos(m))
            continue
        m = LOADED_RE.search(ln)
        if m:
            out["loaded_checkpoint"] = m.group("path")
            continue
        if "[fitckpt]" in ln and any(k in ln for k in REFUSAL_MARKERS):
            out["refusals"].append(ln.strip())
    if out["resume_line"] is None:
        out["restart_from_zero"] = True
        out["reason"] = ("no '[fitckpt] ... RESUMING from checkpoint' line "
                         "— the successor started from zero")
    elif int(out["step"] or 0) <= 0:
        out["reason"] = (f"RESUMING line reports optimizer step "
                         f"{out['step']} — not a resume")
    else:
        out["resumed"] = True
    return out


# --------------------------------------------------------------------------- #
# Three-way comparison (step f)                                                 #
# --------------------------------------------------------------------------- #


def _find_object_file(d: Path) -> Path | None:
    if (d / "object.pt").is_file():
        return d / "object.pt"
    marker = d / "COMPLETE"
    if marker.is_file():
        try:
            name = json.loads(marker.read_text()).get("object_file")
            if name and (d / name).is_file():
                return d / name
        except ValueError:
            pass
    cands = sorted(p for p in d.glob("*.pt")
                   if p.is_file() and not p.name.startswith("ckpt"))
    return cands[0] if cands else None


def _flatten_tensors(obj, prefix: str, out: dict) -> None:
    import torch

    if isinstance(obj, torch.Tensor):
        if torch.is_floating_point(obj):
            out[prefix] = obj.detach().cpu()
        return
    if isinstance(obj, torch.nn.Module):
        _flatten_tensors(obj.state_dict(), prefix, out)
        return
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _flatten_tensors(obj[k], f"{prefix}/{k}", out)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten_tensors(v, f"{prefix}/{i}", out)
        return
    # scalars / strings / other metadata: not parameters


def load_param_leaves(d) -> dict | None:
    """Every float tensor leaf of the dir's object blob, keyed by sorted
    path; None when no object file is present."""
    import torch

    p = _find_object_file(Path(d))
    if p is None:
        return None
    blob = torch.load(str(p), map_location="cpu", weights_only=False)
    leaves: dict = {}
    _flatten_tensors(blob, "", leaves)
    return leaves


def param_distance(a: dict | None, b: dict | None) -> tuple[float, bool]:
    """(L2 distance over the concatenation of all leaves, bitwise-equal).
    A missing blob, a key-set difference or a shape difference is an
    infinite distance (never a silent partial compare)."""
    import torch

    if a is None or b is None:
        return math.inf, False
    if set(a) != set(b):
        return math.inf, False
    sq = 0.0
    bitwise = True
    for k in sorted(a):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            return math.inf, False
        if not torch.equal(x, y):
            bitwise = False
        d = (x.double() - y.double()).flatten()
        sq += float(torch.dot(d, d))
    return math.sqrt(sq), bitwise


def _curve_points(values) -> dict[int, float] | None:
    """A curve as ``{index: value}``: a LIST is indexed from 0 (the fit
    trainers' ``heldout_curve`` layout — index 0 the pre-training value,
    index e+1 the value after epoch e); a dict keeps its integer keys."""
    if isinstance(values, list):
        return {i: float(v) for i, v in enumerate(values)}
    if isinstance(values, dict) and values:
        try:
            return {int(k): float(v) for k, v in values.items()}
        except (TypeError, ValueError):
            return None
    return None


def _curve_from_json(payload, arm: str | None = None) -> dict[int, float] | None:
    """A held-out curve out of a json payload: a top-level ``heldout_curve``
    (the drill's own heldout_curve.json), the fits driver's sidecar
    ``fit_record.heldout_curve`` (train_budget_meta.json — the FULL curve
    even for a resumed cell, since the checkpoint carries it), or a fit
    report's ``arms.<arm>.heldout_curve``."""
    if not isinstance(payload, dict):
        return None
    c = _curve_points(payload.get("heldout_curve"))
    if c is not None:
        return c
    fr = payload.get("fit_record")
    if isinstance(fr, dict):
        c = _curve_points(fr.get("heldout_curve"))
        if c is not None:
            return c
    arms = payload.get("arms")
    if isinstance(arms, dict):
        keys = [arm] if arm and arm in arms else sorted(arms)
        for k in keys:
            rec = arms.get(k)
            if isinstance(rec, dict):
                c = _curve_points(rec.get("heldout_curve"))
                if c is not None:
                    return c
    return None


def _curve_from_metrics(path: Path, arm: str | None) -> dict[int, float] | None:
    """The held-out curve out of a fit's ``metrics.jsonl``: rows carry
    ``<arm>/heldout_loss`` WITH an ``epoch`` field (0-based; stored at index
    epoch+1 so it aligns with the ``heldout_curve`` layout) and one
    ``<arm>/heldout_loss_init`` row (index 0). With ``arm`` None the single
    ``*/heldout_loss`` prefix among the epoch-stamped rows is used; two
    prefixes are ambiguous and yield None. A resumed cell's metrics file holds only the successor
    process's rows — the sidecar's ``fit_record.heldout_curve`` is the full
    curve and is preferred by :func:`load_heldout_curve`."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    rows = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    if arm is None:
        # the curve rows are the ones stamped with an epoch (a mid-fit
        # probe's ``probe/heldout_loss`` row carries a step, not an epoch)
        prefixes = sorted({k[:-len("/heldout_loss")] for r in rows
                           if isinstance(r, dict) and r.get("epoch") is not None
                           for k in r if k.endswith("/heldout_loss")})
        if len(prefixes) != 1:
            return None
        arm = prefixes[0]
    key, init_key = f"{arm}/heldout_loss", f"{arm}/heldout_loss_init"
    out: dict[int, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        if init_key in r:
            out[0] = float(r[init_key])
        if key in r and r.get("epoch") is not None:
            out[int(r["epoch"]) + 1] = float(r[key])
    return out or None


def load_heldout_curve(d, arm: str | None = None) -> dict[int, float] | None:
    """The held-out curve of a cell dir as ``{index: value}`` (index 0 the
    pre-training value, e+1 after epoch e): `heldout_curve.json` first,
    then any json carrying a curve (:func:`_curve_from_json` — the fits
    driver's train_budget_meta.json ``fit_record.heldout_curve`` included),
    then the ``<arm>/heldout_loss`` rows of `metrics.jsonl`. An earlier
    drill found NO curve because the loader knew neither the sidecar path
    nor the arm-prefixed metrics key."""
    d = Path(d)
    cands = [d / "heldout_curve.json"] + sorted(
        p for p in d.glob("*.json") if p.name != "heldout_curve.json")
    for p in cands:
        if not p.is_file():
            continue
        try:
            payload = json.loads(p.read_text())
        except ValueError:
            continue
        curve = _curve_from_json(payload, arm)
        if curve is not None:
            return curve
    metrics = d / "metrics.jsonl"
    if metrics.is_file():
        return _curve_from_metrics(metrics, arm)
    return None


def _argmin(c: dict[int, float]) -> int:
    return min(sorted(c), key=lambda k: c[k])


def curve_distance(a: dict | None, b: dict | None) -> dict:
    """Two curves compared: ``min_absdiff`` (the gated quantity — the
    distance between the best held-out values), ``maxabs`` over the epochs
    BOTH cover (diagnostic), the coverage (``n_common``, ``only_a``,
    ``only_b`` counts), the minima with their indices, and ``bitwise``
    (same index set, equal values). A missing or empty curve is infinite
    everywhere, never a silent pass."""
    if not a or not b:
        return {"min_absdiff": math.inf, "maxabs": math.inf, "n_common": 0,
                "only_a": len(a or {}), "only_b": len(b or {}),
                "min_a": None, "min_b": None, "argmin_a": None,
                "argmin_b": None, "bitwise": False}
    common = sorted(set(a) & set(b))
    ia, ib = _argmin(a), _argmin(b)
    return {
        "min_absdiff": abs(a[ia] - b[ib]),
        "maxabs": (max(abs(a[k] - b[k]) for k in common) if common
                   else math.inf),
        "n_common": len(common),
        "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
        "min_a": a[ia], "min_b": b[ib], "argmin_a": ia, "argmin_b": ib,
        "bitwise": set(a) == set(b) and all(a[k] == b[k] for k in common),
    }


def _within_floor(d3: float, d2: float, floor: float = COMPARE_FLOOR) -> bool:
    return d3 <= max(2.0 * d2, floor)


def _complete_structure(d: Path) -> list | None:
    marker = d / "COMPLETE"
    if not marker.is_file():
        return None
    try:
        files = json.loads(marker.read_text()).get("files")
    except ValueError:
        return None
    return sorted(files) if isinstance(files, dict) else None


def three_way_compare(a_dir, a2_dir, a3_dir, *, arm: str | None = None) -> dict:
    """A = uninterrupted, A' = second uninterrupted same-seed twin (the
    nondeterminism floor), A'' = interrupted + resumed. Pass iff
    dist(A'', A) <= max(2 * dist(A', A), 1e-9) on the parameters
    (:data:`COMPARE_RULE`), |min(A'') - min(A)| <= max(2 * |min(A') -
    min(A)|, 1e-3) on the held-out curve's best value (:data:`CURVE_RULE`;
    the max-abs over the common epochs is reported, not gated), AND the two
    COMPLETE markers list the SAME file set (``complete_structure.match``
    is True — a resumed cell that published more, fewer or differently
    named files is not the same publication, however close its
    parameters); bitwise flags reported alongside."""
    la, la2, la3 = (load_param_leaves(x) for x in (a_dir, a2_dir, a3_dir))
    d2, bw2 = param_distance(la2, la)
    d3, bw3 = param_distance(la3, la)
    params = {"dist_a2_a": d2, "dist_a3_a": d3, "bitwise_a2_a": bw2,
              "bitwise_a3_a": bw3,
              "n_leaves": (len(la) if la is not None else 0),
              "n_elements": (sum(int(t.numel()) for t in la.values())
                             if la else 0),
              "object_found": {"a": la is not None, "a2": la2 is not None,
                               "a3": la3 is not None},
              "pass": bool(math.isfinite(d3) and _within_floor(d3, d2))}
    ca, ca2, ca3 = (load_heldout_curve(x, arm) for x in (a_dir, a2_dir, a3_dir))
    c2, c3 = curve_distance(ca2, ca), curve_distance(ca3, ca)
    curve = {"len_a": (len(ca) if ca is not None else None),
             "len_a2": (len(ca2) if ca2 is not None else None),
             "len_a3": (len(ca3) if ca3 is not None else None),
             "curve_found": {"a": ca is not None, "a2": ca2 is not None,
                             "a3": ca3 is not None},
             # the gated quantity: the best held-out value
             "min_a": c2["min_b"], "min_a2": c2["min_a"], "min_a3": c3["min_a"],
             "argmin_a": c2["argmin_b"], "argmin_a2": c2["argmin_a"],
             "argmin_a3": c3["argmin_a"],
             "min_absdiff_a2_a": c2["min_absdiff"],
             "min_absdiff_a3_a": c3["min_absdiff"],
             "min_tol": CURVE_MIN_TOL,
             # diagnostics: the per-epoch max-abs over the common epochs
             "maxabs_a2_a": c2["maxabs"], "maxabs_a3_a": c3["maxabs"],
             "n_common_a2_a": c2["n_common"], "n_common_a3_a": c3["n_common"],
             "only_a3": c3["only_a"], "only_a_vs_a3": c3["only_b"],
             "bitwise_a2_a": c2["bitwise"], "bitwise_a3_a": c3["bitwise"],
             "pass": bool(math.isfinite(c3["min_absdiff"])
                          and _within_floor(c3["min_absdiff"],
                                            c2["min_absdiff"], CURVE_MIN_TOL))}
    sa, sa3 = _complete_structure(Path(a_dir)), _complete_structure(Path(a3_dir))
    structure = {"a_files": sa, "a3_files": sa3,
                 "match": (sa == sa3) if (sa is not None and sa3 is not None)
                 else None,
                 "reason": (None if (sa is not None and sa3 is not None
                                     and sa == sa3) else
                            "a COMPLETE marker is missing or unreadable"
                            if (sa is None or sa3 is None) else
                            f"published file sets differ: only A "
                            f"{sorted(set(sa) - set(sa3))}, only A'' "
                            f"{sorted(set(sa3) - set(sa))}")}
    return {"rule": {"params": COMPARE_RULE, "curve": CURVE_RULE},
            "dirs": {"a": str(a_dir), "a2": str(a2_dir), "a3": str(a3_dir)},
            "params": params, "curve": curve,
            "complete_structure": structure,
            "pass": bool(params["pass"] and curve["pass"]
                         and structure["match"] is True)}


# --------------------------------------------------------------------------- #
# Orchestration (steps a-g)                                                     #
# --------------------------------------------------------------------------- #


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(
        timespec="seconds")


def _await(client, clock, deadline: float, poll_s: float, probe):
    """Poll `probe()` until it returns a non-None value or the deadline
    passes. Returns (value_or_None, elapsed_s, timed_out)."""
    t0 = clock()
    while True:
        v = probe()
        if v is not None:
            return v, clock() - t0, False
        if clock() >= deadline:
            return None, clock() - t0, True
        client.sleep(poll_s)


def _await_ckpt(client, clock, deadline, poll_s, item, call_id, spawn_t):
    """Wait for ckpt.pt to land with a post-spawn mtime, or for the call to
    end first. Returns evidence dict."""
    path = ckpt_path(item)

    def probe():
        m = client.ckpt_mtime(path)
        if m is not None and m > spawn_t:
            return ("landed", m)
        st = client.state(call_id)
        if st != "running":
            return ("ended", st)
        return None

    v, el, _timed_out = _await(client, clock, deadline, poll_s, probe)
    ev = {"ckpt_path": path, "spawn_time": _iso(spawn_t), "wait_s": round(el, 1)}
    if v is None:
        ev.update(ckpt_landed=False, reason="max_wait elapsed before ckpt.pt appeared")
    elif v[0] == "landed":
        ev.update(ckpt_landed=True, ckpt_mtime=_iso(v[1]),
                  seconds_after_spawn=round(v[1] - spawn_t, 1))
    else:
        ev.update(ckpt_landed=False, call_state=v[1],
                  reason=f"call ended ({v[1]}) before any checkpoint landed")
    return ev


def _await_terminal(client, clock, deadline, poll_s, call_id):
    def probe():
        st = client.state(call_id)
        return st if st != "running" else None

    v, el, timed_out = _await(client, clock, deadline, poll_s, probe)
    return {"state": v, "wall_s": round(el, 1),
            "timed_out_waiting": timed_out}


def _materialize(client, item: tuple, local_dir: Path) -> dict:
    """Bring a COMPLETE cell's artifacts to `local_dir` for the comparison,
    from the CURRENT marker only: every file the marker's ``{name: sha}``
    manifest lists is fetched and its sha256 verified against the marker's
    before anything is written; a missing or mismatching file leaves the
    dir EMPTY and reports why (the compare then reads "no object", never a
    stale or torn artifact). A dir that already exists is refused — a
    drill never compares against a previous run's files."""
    if local_dir.exists():
        return {"materialized": False, "dir": str(local_dir),
                "reason": "comparison dir already exists — a drill never "
                          "compares against a previous run's artifacts"}
    marker = client.read_json(complete_path(item))
    if not marker:
        return {"materialized": False, "dir": str(local_dir),
                "reason": "no COMPLETE marker on the volume"}
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        return {"materialized": False, "dir": str(local_dir),
                "reason": "COMPLETE marker carries no {name: sha256} file "
                          "manifest — nothing can be verified"}
    fetched: dict[str, bytes] = {}
    missing, mismatch = [], []
    for name, sha in files.items():
        b = client.read_bytes(f"{cell_dir_path(item)}/{name}")
        if b is None:
            missing.append(str(name))
            continue
        got = hashlib.sha256(b).hexdigest()
        if got != str(sha):
            mismatch.append(f"{name}: fetched {got[:12]}… != marker "
                            f"{str(sha)[:12]}…")
            continue
        fetched[str(name)] = b
    ev = {"dir": str(local_dir), "fetched": sorted(fetched),
          "missing": missing, "sha_mismatch": mismatch}
    if missing or mismatch:
        return {"materialized": False, **ev,
                "reason": "the marker's file set could not be fetched and "
                          "sha-verified in full"}
    local_dir.mkdir(parents=True, exist_ok=False)
    for name, b in fetched.items():
        (local_dir / name).write_bytes(b)
    (local_dir / "COMPLETE").write_text(json.dumps(marker, indent=1,
                                                  sort_keys=True) + "\n")
    # a fit REPORT's curve, when one exists, is pinned beside the artifacts;
    # otherwise the loader reads the sidecar's fit_record.heldout_curve or
    # the metrics rows (both are in the marker's file set)
    rep = client.read_json(report_path(item))
    curve = _curve_from_json(rep, item[2]) if rep else None
    if curve is not None:
        (local_dir / "heldout_curve.json").write_text(
            json.dumps({"heldout_curve": {str(k): v for k, v in curve.items()},
                        "arm": item[2]}) + "\n")
    return {"materialized": True, "verified": True, **ev,
            "curve_points": (len(curve) if curve is not None else None)}


def run_drill(client, *, items: dict, out_root: str, poll_s: float,
              max_wait_s: float, ckpt_every: int, function_timeout_s: float,
              app: str = "", function: str = "", run_id: str = "",
              twin_client=None, twin_app: str = "",
              clock=None, log=print) -> dict:
    """Steps (a)-(g); returns the verdict dict. `twin_client` / `twin_app`
    name the long-timeout deploy of the same commit the uninterrupted
    twins spawn on (default: `client` / `app` — see the module docstring
    for why that default cannot pass a real drill). `items` maps the four
    ITEM_NAMES to {"item": 7-tuple, "dir": local compare dir} (build_items
    namespaces both by `run_id`; a compare dir that already exists refuses
    before anything is spawned). `clock` defaults to the client's `now()`
    if it has one (fakes), else time.time."""
    fw = _fits_watch()
    assert tuple(fw.RESPAWN_CAUSES) == ENDED_STATES, (
        "drill/watcher disagree on the continuation causes")
    if not app or not function:
        raise ValueError("run_drill needs the drill app and function names: "
                         "they are the spawn pin the manifest rows carry "
                         "(fits_watch.row_pin), and a pin-less row is never "
                         "respawned")
    clock = clock or getattr(client, "now", None) or time.time
    twin_client = twin_client if twin_client is not None else client
    twin_app = twin_app or app
    missing = [n for n in ITEM_NAMES if n not in items]
    if missing:
        raise ValueError(f"items must carry {ITEM_NAMES}; missing {missing}")
    for n in ITEM_NAMES:
        check_item_shape(items[n]["item"])
    roots = [items[n]["item"][5] for n in ITEM_NAMES]
    if len(set(roots)) != len(roots):
        raise ValueError(f"drill items must have distinct out_roots: {roots}")
    existing = [items[n]["dir"] for n in ITEM_NAMES
                if Path(items[n]["dir"]).exists()]
    if existing:
        raise ValueError(
            f"comparison dir(s) already exist: {existing} — a drill never "
            "compares against a previous run's artifacts; pick a new "
            "--run-id (or --compare-root)")

    t_start = clock()
    deadline = t_start + float(max_wait_s)
    steps: dict = {}
    rows: dict = {n: {"item": list(items[n]["item"]), "dir": items[n]["dir"]}
                  for n in ITEM_NAMES}

    def record(name: str, ok: bool, **ev) -> None:
        steps[name] = {"verdict": "PASS" if ok else "FAIL", **ev}
        log(f"[drill] {name}: {'PASS' if ok else 'FAIL'} "
            + json.dumps(ev, default=str)[:400], flush=True)

    def spawn(name: str) -> str:
        t = clock()
        cid = (twin_client if name in TWINS else client).spawn(
            items[name]["item"])
        rows[name].update(call_id=cid, spawn_t=t, spawn_time=_iso(t))
        log(f"[drill] spawned {name}: {cid}", flush=True)
        return cid

    # (a) the deploy: record the deployed function id AND the deployment
    # version (the pin that actually moves on redeploy).
    fid = client.function_id()
    twin_fid = twin_client.function_id()
    dep_at_spawn = _client_deployment(client)
    record("a_deploy", bool(fid) and bool(twin_fid), function_id=fid, app=app,
           function=function, deployment=dep_at_spawn,
           expected_function_timeout_s=function_timeout_s,
           operator_step=(f"SDFBASE_FIT_TIMEOUT={int(function_timeout_s)} "
                          f"modal deploy --name {app or '<drill-app>'} "
                          "scripts/modal/sdfbase_fits.py"),
           twin_app=twin_app, twin_function_id=twin_fid,
           twins_on_long_timeout_app=(twin_client is not client),
           operator_step_twins=(f"modal deploy --name {twin_app} "
                                "scripts/modal/sdfbase_fits.py  (same "
                                "commit, production timeout)"))

    # Twins first: they are the longest-running items and need nothing
    # from the interrupted ones.
    for n in TWINS:
        spawn(n)

    # (b) the timeout item: spawn, wait for its checkpoint to land.
    cid_to = spawn("timeout")
    ev_b = _await_ckpt(client, clock, deadline, poll_s, items["timeout"]["item"],
                       cid_to, rows["timeout"]["spawn_t"])
    record("b_checkpoint_landed", bool(ev_b.get("ckpt_landed")),
           checkpoint_every=ckpt_every, **ev_b)

    # (c2) the cancel item: spawn, wait for its checkpoint, cancel it.
    cid_ca = spawn("cancel")
    ev_ca = _await_ckpt(client, clock, deadline, poll_s, items["cancel"]["item"],
                        cid_ca, rows["cancel"]["spawn_t"])
    cancelled = False
    if ev_ca.get("ckpt_landed"):
        client.cancel(cid_ca)
        cancelled = True
        ev_ca["cancel_time"] = _iso(clock())
    ev_ca_end = _await_terminal(client, clock, deadline, poll_s, cid_ca)
    rows["cancel"]["state"] = ev_ca_end["state"]
    record("c_cancel", bool(cancelled and ev_ca_end["state"] in ENDED_STATES),
           cancelled=cancelled, observed_state=ev_ca_end["state"],
           wall_after_cancel_s=ev_ca_end["wall_s"], **ev_ca)

    # (c1) the timeout item hits the REAL function timeout.
    ev_to = _await_terminal(client, clock, deadline, poll_s, cid_to)
    rows["timeout"]["state"] = ev_to["state"]
    wall_since_spawn = round(clock() - rows["timeout"]["spawn_t"], 1)
    record("c_real_timeout", ev_to["state"] == "timeout",
           observed_state=ev_to["state"], wall_since_spawn_s=wall_since_spawn,
           expected_function_timeout_s=function_timeout_s,
           **({"reason": ("the fit finished before the function timeout — "
                          "use a bigger cell or a shorter "
                          "SDFBASE_FIT_TIMEOUT")}
              if ev_to["state"] == "done" else {}))

    # (d) exactly one successor per ended item, via the PRODUCTION path:
    # the row carries its own spawn pin, the marker verdict is read the way
    # the watcher reads it (existence-only here: the drill records no
    # expect_fingerprint, so a present marker is "complete").
    fn = _ClientFn(client)
    for n in INTERRUPTED:
        state = rows[n].get("state")
        cell = manifest_cell(
            items[n]["item"], app=app, function=function, function_id=fid,
            call_id=rows[n]["call_id"], spawn_time=rows[n]["spawn_time"],
            deployment=dep_at_spawn)
        marker_state, _detail = fw.marker_verdict(
            client.read_bytes(complete_path(items[n]["item"])),
            cell.get("expect_fingerprint"), strict=False)
        ok, why = fw.can_respawn(cell, state or "running", max_successors=1,
                                 marker=marker_state)
        live_dep = _client_deployment(client)
        ok_v, why_v = fw.same_app_version(cell, fn, live_deployment=live_dep)
        ev = {"predecessor_call_id": rows[n]["call_id"],
              "predecessor_state": state, "marker_state": marker_state,
              "can_respawn": ok, "can_respawn_reason": why,
              "same_app_version": ok_v, "same_app_version_reason": why_v,
              "live_deployment": live_dep}
        if ok and ok_v:
            rec = fw.spawn_successor(cell, state, fn, _iso(clock()),
                                     live_deployment=live_dep)
            rows[n]["successor"] = rec
            rows[n]["successor_call_id"] = rec["call_id"]
            again, why_again = fw.can_respawn(cell, state, max_successors=1,
                                              marker=marker_state)
            ev.update(successor=rec, n_successors=len(cell["successors"]),
                      cap_honored=(not again), cap_reason=why_again,
                      cell_status=cell["status"])
            record(f"d_successor_{n}",
                   len(cell["successors"]) == 1 and not again, **ev)
        else:
            record(f"d_successor_{n}", False, **ev)

    # (e) successors end `done` + COMPLETE, and their lines prove a resume.
    for n in INTERRUPTED:
        scid = rows[n].get("successor_call_id")
        if not scid:
            record(f"e_resume_{n}", False,
                   reason="no successor was spawned (step d failed)")
            continue
        end = _await_terminal(client, clock, deadline, poll_s, scid)
        res = client.result(scid) or {}
        lines = list(res.get("fitckpt_lines") or [])
        logs = client.logs(scid) or ""
        lines += [ln for ln in logs.splitlines() if "[fitckpt]" in ln]
        proof = resume_proof(lines)
        complete = client.read_json(complete_path(items[n]["item"])) is not None
        record(f"e_resume_{n}",
               bool(end["state"] == "done" and complete and proof["resumed"]),
               successor_call_id=scid, successor_state=end["state"],
               successor_wall_s=end["wall_s"], complete_marker=complete,
               result_status=res.get("status"), result_rc=res.get("rc"),
               resume_proof=proof)

    # (f) twins COMPLETE, then the three-way comparison per interrupted item.
    for n in TWINS:
        p = complete_path(items[n]["item"])
        v, el, timed_out = _await(client, clock, deadline, poll_s,
                                  lambda p=p: client.read_json(p))
        rows[n]["complete"] = v is not None
        rows[n]["state"] = twin_client.state(rows[n]["call_id"])
        rows[n]["complete_wait_s"] = round(el, 1)
    twins_ok = all(rows[n]["complete"] for n in TWINS)
    record("f_twins_complete", twins_ok,
           **{n: {"call_id": rows[n]["call_id"], "state": rows[n]["state"],
                  "complete": rows[n]["complete"]} for n in TWINS})
    mat = {}
    for n in ITEM_NAMES:
        mat[n] = _materialize(client, items[n]["item"], Path(items[n]["dir"]))
    record("f_materialized",
           all(bool(m.get("materialized")) for m in mat.values()), **mat)
    arm = items["twin_a"]["item"][2]
    for n in INTERRUPTED:
        cmp = three_way_compare(items["twin_a"]["dir"], items["twin_a2"]["dir"],
                                items[n]["dir"], arm=arm)
        record(f"f_compare_{n}", cmp["pass"], **cmp)

    # (g) the verdict.
    overall = all(s["verdict"] == "PASS" for s in steps.values())
    t_end = clock()
    verdict = {
        "drill": "continuation", "plan_ref": "sdf-base-steering-r2 §3.2.4",
        "verdict": "PASS" if overall else "FAIL", "pass": overall,
        "started_at": _iso(t_start), "finished_at": _iso(t_end),
        "elapsed_s": round(t_end - t_start, 1),
        "function_id": fid, "run_id": run_id,
        "twin_app": twin_app, "twin_function_id": twin_fid,
        "params": {"app": app, "function": function, "out_root": out_root,
                   "twin_app": twin_app,
                   "poll_s": poll_s, "max_wait_s": max_wait_s,
                   "checkpoint_every": ckpt_every,
                   "function_timeout_s": function_timeout_s},
        "items": rows, "steps": steps,
        "failed_steps": [k for k, s in steps.items() if s["verdict"] != "PASS"],
        "notes": ("§3.2.4h: derive production --checkpoint-every (<= ~250 "
                  "optimizer steps) and the expected continuation count per "
                  "fit from the fit telemetry, not from this drill."),
    }
    log(f"[drill] VERDICT {verdict['verdict']} "
        f"({len(verdict['failed_steps'])} failed step(s): "
        f"{verdict['failed_steps']})", flush=True)
    return verdict


def write_verdict(verdict: dict, out) -> Path:
    p = Path(out).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    tmp.replace(p)
    return p


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default=None,
                    help="the THROWAWAY drill app name the wave-A commit was "
                         "deployed under (never the production app); "
                         "required unless --compare-only")
    ap.add_argument("--twin-app", default=None,
                    help="the SECOND throwaway app (same commit, production "
                         "timeout) the uninterrupted twins A / A' run on; "
                         "default: the drill app, which cannot pass once "
                         "the fit outlasts the drill timeout")
    ap.add_argument("--function", default="fit_h200_std",
                    help="fit_h200_std (gemma) or fit_h200_qwen")
    ap.add_argument("--model", default="gemma4e4b_base")
    ap.add_argument("--si", default="constv2-mix-1k")
    ap.add_argument("--arm", default="psr_ce_all")
    ap.add_argument("--config-sha", default=None,
                    help="expect_sha: sha256(configs/<fits config> bytes)[:16] "
                         "at the deployed commit — the same value fits-spawn "
                         "computes; required unless --compare-only")
    ap.add_argument("--compare-only", metavar="RUN_COMPARE_DIR", default=None,
                    help="skip every Modal step: run ONLY the three-way "
                         "comparison (step f) on a previous run's materialized "
                         "<compare-root>/<run-id>/{twin_a,twin_a2,timeout,"
                         "cancel} dirs and write that verdict to --out")
    ap.add_argument("--out-root", default="out/drill",
                    help="volume root; each item gets <out-root>/<run-id>/<name>")
    ap.add_argument("--run-id", default=None,
                    help="this run's namespace token under --out-root and "
                         "--compare-root (default: the UTC start time); a "
                         "compare dir that already exists refuses")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--function-timeout", type=float, default=600.0,
                    help="the SDFBASE_FIT_TIMEOUT the drill app was deployed "
                         "with (seconds)")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--max-wait", type=float, default=5400.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare-root",
                    default="$STAGE/sdfbase/drill",
                    help="local dir where the four cells' artifacts land, "
                         "under <compare-root>/<run-id>/<name>")
    ap.add_argument("--out", default="$STAGE/sdfbase/drill_verdict.json",
                    help="the verdict file (default under the staging "
                         "directory, $STAGE)")
    return ap


def expand_site_paths(args) -> None:
    """--compare-root and --out with their site placeholder ($STAGE, the
    staging directory) expanded through the site config
    (scripts/site_paths.py); an unset one refuses by name."""
    if str(HERE.parent) not in sys.path:
        sys.path.append(str(HERE.parent))
    from site_paths import site_path
    args.compare_root = str(site_path(args.compare_root, "--compare-root"))
    args.out = str(site_path(args.out, "--out"))


def require_torch() -> str | None:
    """The comparison (step f) loads object blobs with torch; refuse at
    START when the interpreter has none, instead of crashing after the GPU
    steps ran (an earlier run was invoked with the modal venv and died at
    its final compare). Returns the problem, or None."""
    if importlib.util.find_spec("torch") is None:
        return (f"torch is not importable from {sys.executable} — the drill's "
                "three-way comparison needs it; run with the repo venv "
                "(.venv/bin/python), not the modal venv")
    return None


def compare_only(run_dir, *, arm: str | None, out) -> dict:
    """Step (f) alone on a materialized run dir: twin_a / twin_a2 / each
    interrupted item; writes and returns the compare-only verdict."""
    run_dir = Path(run_dir).expanduser()
    dirs = {n: run_dir / n for n in ITEM_NAMES}
    missing = [n for n, d in dirs.items() if not d.is_dir()]
    if missing:
        raise ValueError(f"{run_dir}: missing materialized item dir(s) "
                         f"{missing} (expected {list(ITEM_NAMES)})")
    compares = {n: three_way_compare(dirs["twin_a"], dirs["twin_a2"], dirs[n],
                                     arm=arm) for n in INTERRUPTED}
    verdict = {"drill": "continuation", "compare_only": True,
               "run_dir": str(run_dir), "arm": arm,
               "compares": compares,
               "pass": all(c["pass"] for c in compares.values()),
               "finished_at": _iso(time.time())}
    verdict["verdict"] = "PASS" if verdict["pass"] else "FAIL"
    p = write_verdict(verdict, out)
    for n, c in compares.items():
        print(f"[drill] compare {n}: params {'PASS' if c['params']['pass'] else 'FAIL'} "
              f"(dist {c['params']['dist_a3_a']:.3e} vs twin {c['params']['dist_a2_a']:.3e}); "
              f"curve {'PASS' if c['curve']['pass'] else 'FAIL'} "
              f"(|dmin| {c['curve']['min_absdiff_a3_a']:.3e} vs twin "
              f"{c['curve']['min_absdiff_a2_a']:.3e}, tol {CURVE_MIN_TOL:g}; "
              f"max-abs {c['curve']['maxabs_a3_a']:.3e} vs twin "
              f"{c['curve']['maxabs_a2_a']:.3e} over {c['curve']['n_common_a3_a']} "
              f"common epochs — diagnostic); structure "
              f"{c['complete_structure']['match']}", flush=True)
    print(f"[drill] compare-only verdict {verdict['verdict']} written: {p}",
          flush=True)
    return verdict


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    expand_site_paths(args)
    problem = require_torch()
    if problem:
        print(f"[drill] REFUSED: {problem}", file=sys.stderr, flush=True)
        return 2
    if args.compare_only:
        v = compare_only(args.compare_only, arm=args.arm, out=args.out)
        return 0 if v["pass"] else 1
    if not args.app or not args.config_sha:
        print("[drill] REFUSED: --app and --config-sha are required (unless "
              "--compare-only)", file=sys.stderr, flush=True)
        return 2
    print("[drill] expects these operator steps done first (plan §3.2.4a):",
          flush=True)
    print(f"[drill]   SDFBASE_FIT_TIMEOUT={int(args.function_timeout)} "
          f"modal deploy --name {args.app} scripts/modal/sdfbase_fits.py",
          flush=True)
    print(f"[drill]   pool/{args.model}/pairs_{args.si}.jsonl (+ tokcache) "
          f"and pool/{args.model}/pairs_constv2-heldout.jsonl staged on the "
          "twosi-fits volume", flush=True)
    print(f"[drill]   --config-sha {args.config_sha} = the deployed commit's "
          "config bytes (a mismatch makes every fit refuse in-container)",
          flush=True)
    run_id = args.run_id or default_run_id()
    print(f"[drill] run id {run_id}: items under {args.out_root}/{run_id}/, "
          f"comparison under {args.compare_root}/{run_id}/", flush=True)
    items = build_items(model=args.model, si=args.si, arm=args.arm,
                        config_sha=args.config_sha, out_root=args.out_root,
                        checkpoint_every=args.checkpoint_every,
                        compare_root=args.compare_root, run_id=run_id,
                        overwrite=args.overwrite)
    twin_app = args.twin_app or args.app
    print(f"[drill]   twins on {twin_app}"
          + (" (the drill app: only passes if the fit finishes under the "
             "drill timeout)" if twin_app == args.app else
             " — modal deploy --name " + twin_app + " scripts/modal/"
             "sdfbase_fits.py at the SAME commit, production timeout"),
          flush=True)
    client = RealModalClient(args.app, args.function)
    twin_client = (RealModalClient(twin_app, args.function)
                   if twin_app != args.app else client)
    verdict = run_drill(client, items=items, out_root=args.out_root,
                        poll_s=args.poll, max_wait_s=args.max_wait,
                        ckpt_every=args.checkpoint_every,
                        function_timeout_s=args.function_timeout,
                        app=args.app, function=args.function, run_id=run_id,
                        twin_client=twin_client, twin_app=twin_app)
    p = write_verdict(verdict, args.out)
    print(f"[drill] verdict written: {p}", flush=True)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
