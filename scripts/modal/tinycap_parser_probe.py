#!/usr/bin/env python
"""The tool-call parser probe on Modal — `check_tool_parser_offline`
(CPU, the serving image, the REAL vLLM 0.25.1 ``qwen3_xml`` parser over the pinned tokenizer; the
definite fixtures asserted exactly, the either-of fixtures asserted to ONE admitted reading,
the edges recorded)
and `probe_tool_parser` (GPU: the production standup of the row's ``none`` cell under
``--tool-calls --manifest <family full22 manifest>``, the 24 pinned BFCL items with the
same-request ``return_token_ids`` raw capture, detokenize + replay through a fresh parser).
The pure rules live in `scripts/bfcl_parser_probe.py` (READ FROM THE STAGED TREE); this file is
the Modal plumbing. Metered as stage kind `parser-probe` (unit `parser:<family>`).

    TINYCAP_APP_NAME=tinycap-bcbw-parser-qwen35-<sha7> ~/venvs/modal/bin/python -m modal run --detach \\
        scripts/modal/tinycap_parser_probe.py --sha <40-hex> --family qwen3_5
    TINYCAP_APP_NAME=tinycap-bcbw-parser-nemotronh-<sha7> ~/venvs/modal/bin/python -m modal run --detach \\
        scripts/modal/tinycap_parser_probe.py --sha <40-hex> --family nemotron_h

Writes ``/vol/tinycap/parser_probe/<model_key>/<sha7>/{parser_check,parser_probe,verdict}.json``.
The combined verdict is PASS only when BOTH halves PASS (`bfcl_parser_probe.combined_verdict`);
`on_pass_actions(family)` names what the orchestrator lands in ONE change. Nothing here edits
the table, the rows or the roster.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import modal

HERE = Path(__file__).resolve().parent


def _repo_local() -> Path:
    """The checkout root on the LOCAL side (this file lives at scripts/modal/). Inside the Modal
    container the script is mounted flat at ``/root/<name>.py`` with no such ancestor —
    ``HERE.parents[1]`` raised ``IndexError: 1`` there and crash-looped every container of this
    app (confirmed live) — so walk up to the first ancestor that IS a checkout and fall
    back to the script's own directory: the container never reads the local tree (its
    subprocesses run the STAGED one from the volume) and reaches ``sisteer`` through the
    launcher image's ``add_local_python_source``. The same rule as ``tinycap_served._repo_local``."""
    for parent in (HERE, *HERE.parents):
        try:
            if (parent / "src" / "sisteer").is_dir() and (parent / "scripts" / "modal").is_dir():
                return parent
        except OSError:            # an unreadable ancestor is not a checkout
            continue
    return HERE


REPO_LOCAL = _repo_local()
if str(REPO_LOCAL / "src") not in sys.path:
    sys.path.insert(0, str(REPO_LOCAL / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402

APP_NAME_ENV = C.MODAL_APP_NAME_ENV
APP_NAME = os.environ.get(APP_NAME_ENV, "").strip()
_is_local = getattr(modal, "is_local", None)
if not APP_NAME and _is_local is not None and _is_local():
    raise SystemExit(f"[parser-probe] {APP_NAME_ENV} is not set: run as {APP_NAME_ENV}=tinycap-bcbw-parser-<family>-<sha7> "
                     "modal run --detach scripts/modal/tinycap_parser_probe.py --sha … --family …")
import tinycap_served as T  # noqa: E402  (the launcher: its images, volume, standup composition, staged-tree resolver)

#: Confirmed by executing both scripts at ``/root/<name>.py`` with no checkout on sys.path:
#: Modal mounts each function's OWN entrypoint, and the shared image ships
#: ``tinycap_images.py`` + ``sisteer`` but NOT ``tinycap_served.py`` — so the line above raised
#: ``ModuleNotFoundError`` inside every container of this app. The launcher module is mounted at
#: ``/root`` beside the entrypoint on EVERY image this script's functions run on; the inventory a
#: container root holds is ``container_inventory()`` (the launcher's declared files + these), and
#: the regression test rebuilds a root from exactly that inventory with the checkout's
#: ``scripts/modal`` OFF sys.path.
LAUNCHER_REMOTE = "/root/tinycap_served.py"
CONTAINER_FILES: Dict[str, str] = {LAUNCHER_REMOTE: str(Path(T.__file__).resolve())}


def with_container_files(img):
    """``img`` + every :data:`CONTAINER_FILES` mount (a mount, not a copy layer: the launcher
    changes every commit and nothing builds on top of it here)."""
    for remote, local in CONTAINER_FILES.items():
        img = img.add_local_file(local, remote)
    return img


def container_inventory() -> Dict[str, str]:
    """``{remote path under /root: local source}`` of every file a container of this app holds
    beside its entrypoint — the launcher's own image files plus this script's mounts."""
    return {**dict(T.IMAGE_LOCAL_FILES), **CONTAINER_FILES}


IMAGE = with_container_files(T.image)
IMAGE_NEMOTRON = with_container_files(getattr(T, "image_nemotron", None) or T.image)

app = modal.App(APP_NAME or "tinycap-parser-probe-remote-import")

PROBE_MODULE_REL = "scripts/bfcl_parser_probe.py"
BFCL_GEOMETRY_KEY = f"capability:{C.GEOMETRY_CLASS_AGENT}"
STANDUP_WAIT_S = 45 * 60
# The probe's two halves are PLAIN functions in sisteer.capeval.tinycap.parser_probe_live (this module builds a
# modal.App at import, so the launcher's in-container canary cannot import it); the names below stay as this module's spellings
from sisteer.capeval.tinycap import parser_probe_live as PL  # noqa: E402
REQUEST_TIMEOUT_S = PL.REQUEST_TIMEOUT_S
REQUEST_CONCURRENCY = PL.REQUEST_CONCURRENCY
ITEMS_DUMP_TIMEOUT_S = PL.ITEMS_DUMP_TIMEOUT_S
#: The snippet that dumps the probe items under the CLIENT venv (inspect_evals lives there, never
#: in the serving interpreter — the two interpreters are never merged): the pinned ids'
#: messages and their tools as NATIVE OpenAI tools through inspect_evals' own ToolInfo conversion.
ITEMS_SNIPPET = PL.ITEMS_SNIPPET


def _probe_module(tree: str):
    """`scripts/bfcl_parser_probe.py` FROM THE STAGED TREE (its pins and fixtures are the tree's)."""
    path = Path(tree) / PROBE_MODULE_REL
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the staged tree carries no parser-probe module — stage a tree at or after the lane's landing")
    return T._load_tree_module(tree, PROBE_MODULE_REL, "bfcl_parser_probe_staged")


vllm_tool_parser = PL.vllm_tool_parser


chat_request = PL.chat_request


def parser_reading(parser, text: str, tools: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """parser_probe_live.parser_reading over THIS module's `chat_request` seam (resolved at call time, so the standalone tests
    replace it by name without vLLM installed)."""
    return PL.parser_reading(parser, text, tools, request=chat_request)


_tokenizer = PL.pinned_tokenizer


def _out_dir(P, row: C.ModelRow, sha: str) -> Path:
    d = Path(P.probe_out_dir(row.model_key, sha))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    T.VOL.commit()


def _check_offline(sha: str, family: str) -> dict:
    rec: Dict[str, Any] = {"name": "parser-check-offline", "family": family, "sha": sha[:7]}
    try:
        tree_path, tree_sha = T.resolve_staged_tree(sha)
    except (T.NoStagedTree, T.TreeShaMissing, T.TreeShaMismatch) as e:
        return {**rec, "status": "NO_TREE", "error": f"{type(e).__name__}: {e}"}
    tree = str(tree_path)
    P = _probe_module(tree)
    if family not in P.PROBE_ROWS:
        return {**rec, "status": "UNKNOWN_FAMILY", "error": f"{family!r} not in {sorted(P.PROBE_ROWS)}"}
    row = C.model_row(P.PROBE_ROWS[family])
    rec.update(model=row.slug, model_key=row.model_key, tree_sha=tree_sha)
    return offline_check_for(P, row, tree, sha, family, rec)


def offline_check_for(P, row: C.ModelRow, tree: str, sha: str, family: str, rec: Optional[Dict[str, Any]] = None) -> dict:
    """THE OFFLINE HALF for ``row`` (parser_probe_live.offline_check_for), written to ``probe_out_dir(row.model_key, sha)/parser_check.json``."""
    # this module's seams (the standalone tests monkeypatch them by name): the tokenizer, the real parser factory, the reading
    try:
        tok = _tokenizer(row)
    except Exception as e:  # noqa: BLE001
        return {**dict(rec or {"name": "parser-check-offline", "family": family, "sha": sha[:7], "model": row.slug, "model_key": row.model_key}),
                "status": "TOKENIZER_UNAVAILABLE", "error": f"{type(e).__name__}: {e} — stage the model into the volume HF cache first"}
    doc = PL.offline_check_for(P, row, tree, sha, family, tokenizer=tok, rec=rec, tool_parser=vllm_tool_parser, reading=parser_reading)
    if doc.get("status") == "OK":
        _write(_out_dir(P, row, sha) / "parser_check.json", doc)
    return doc


_post = PL.post_json


def _base_route_vmodel(endpoint: Mapping[str, Any]) -> str:
    from sisteer.capeval.serving import proxy as px
    table = px.load_routing_table(Path(str(endpoint["routing_table"])))
    for r in table["routes"]:
        if r.cell_id == "none":
            return str(r.virtual_model)
    raise LookupError("the routing table carries no `none` route")


def _probe_live(sha: str, family: str) -> dict:
    from sisteer.capeval.tinycap import rung_probe as RP
    rec: Dict[str, Any] = {"name": "parser-probe-live", "family": family, "sha": sha[:7]}
    try:
        tree_path, tree_sha = T.resolve_staged_tree(sha)
    except (T.NoStagedTree, T.TreeShaMissing, T.TreeShaMismatch) as e:
        return {**rec, "status": "NO_TREE", "error": f"{type(e).__name__}: {e}"}
    tree = str(tree_path)
    P = _probe_module(tree)
    if family not in P.PROBE_ROWS:
        return {**rec, "status": "UNKNOWN_FAMILY", "error": f"{family!r} not in {sorted(P.PROBE_ROWS)}"}
    row = C.model_row(P.PROBE_ROWS[family])
    rec.update(model=row.slug, model_key=row.model_key, tree_sha=tree_sha)
    try:
        pin = P.load_probe_pin(tree)
    except (P.ParserProbeError, C.SubsetPinError) as e:
        return {**rec, "status": "PIN_INVALID", "error": str(e)}
    out_dir = _out_dir(P, row, sha)

    def done(doc: Dict[str, Any]) -> Dict[str, Any]:
        """EVERY outcome from here on is filed as `parser_probe.json` (code review: a non-OK
        live probe used to survive only in the container log and `verdict.json`'s status)."""
        _write(out_dir / "parser_probe.json", doc)
        return doc

    try:
        cfg_path, overlaid = T.overlaid_config(row, tree, out_dir)
    except (KeyError, OSError, ValueError) as e:
        return done({**rec, "status": "CONFIG_OVERLAY_FAILED", "error": f"{type(e).__name__}: {e}"})
    # the BFCL leg's pass-1 geometry for THIS model (the same resolution every launch makes)
    sg = T._geometry_module(tree)
    try:
        geom = sg.pass1_geometry(BFCL_GEOMETRY_KEY, model_key=row.model_key)
    except Exception as e:  # noqa: BLE001
        return done({**rec, "status": "GEOMETRY_REFUSED", "error": f"{type(e).__name__}: {e}"})
    manifest = P.PROBE_MANIFESTS[family]
    if not (Path(tree) / manifest).is_file():
        return done({**rec, "status": "NO_MANIFEST", "error": f"{manifest} is not on the staged tree"})
    extra = C.standup_extra_argv(tool_calls=True, thinking=C.THINKING_OFF, family=row.family, manifest=manifest)
    mamba_contract = (f"{tree}/{T._mamba_backend().CONTRACT_RELPATH}" if row.family == T._mamba_backend().NEMOTRON_FAMILY else None)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/src:{tree}/scripts"
    env[C.LENS_STACKED_REDUCTIONS_ENV] = C.REDUCTION_MODE_LAUNCHER_DEFAULT
    srv = Path(f"/tmp/parser_probe_{row.slug}")
    if srv.exists():
        shutil.rmtree(srv)
    srv.mkdir(parents=True)
    stop = srv / "STOP"
    argv = T._standup_cmd(tree, cfg_path, "none", row.objects_dir, srv, stop, geom, row.top_logprobs, extra,
                          mamba_contract=mamba_contract)
    rec.update(standup_argv=argv, geometry={"max_model_len": geom.max_model_len, "max_num_seqs": geom.max_num_seqs}, manifest=manifest)
    print(f"[parser-probe] standing up {row.slug} none for the parser probe: {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(argv, env=env, stdout=(srv / "standup.log").open("w"), stderr=subprocess.STDOUT)
    ready = T._wait_for(srv / "endpoint.json", proc, STANDUP_WAIT_S)
    try:
        if not ready:
            log_text = T._tail(srv / "standup.log", 200_000) or ""
            # a KV-coverage refusal counts only from a standup that EXITED (the rung probe's rule);
            # a standup still alive at the deadline is a timeout, recorded as such
            exited = proc.poll() is not None
            rec.update(status="STANDUP_FAILED", standup_tail=log_text[-3000:], standup_exited=exited,
                       engine_tail=T.engine_core_tail(T._tail(srv / "vllm_server.log", 2_000_000)),
                       kv_refusal=RP.kv_coverage_refusal(log_text) if exited else None)
            return done(rec)
        endpoint = json.loads((srv / "endpoint.json").read_text())
        rec = live_probe_against(P, row, tree, endpoint, env, pin, rec)
        if rec.get("status") not in (None, "OK"):
            return done(rec)
    finally:
        T._tear_down_standup(stop, proc)
        for name in ("standup.log", "vllm_server.log"):
            if (srv / name).exists():
                (out_dir / f"probe_{name}").write_text((srv / name).read_text(errors="replace")[-2_000_000:])
    rec["wall_clock_s"] = round(time.time() - t0, 1)
    return done(rec)


def live_probe_against(P, row: C.ModelRow, tree: str, endpoint: Mapping[str, Any], env: Mapping[str, str], pin: Mapping[str, Any],
                       rec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """THE LIVE HALF against an EXISTING standup (parser_probe_live.live_probe_against with this module's seams: the launcher's full22
    endpoint gate, this module's `_post` / `_base_route_vmodel` / `_tokenizer` / `subprocess.run`, so the standalone tests' monkeypatches
    bind as before). The caller writes the record."""
    def _refusal(row_, endpoint_):
        return T.full22_endpoint_refusal(row_, endpoint_, tree=tree)
    # the tokenizer is read AFTER the endpoint gates and the item dump inside the pure function — pass the seam, not the object
    class _LazyTok:
        _tok = None
        def __getattr__(self, name):
            if _LazyTok._tok is None:
                _LazyTok._tok = _tokenizer(row)
            return getattr(_LazyTok._tok, name)
    return PL.live_probe_against(P, row, tree, endpoint, env, pin, endpoint_refusal=_refusal, tokenizer=_LazyTok(), post=_post,
                                 run=subprocess.run, client_py=T.CLIENT_PY, rec=rec, log=lambda m: print(m, flush=True),
                                 tool_parser=vllm_tool_parser, reading=parser_reading, vmodel_of=_base_route_vmodel)


def _combine(P, row: C.ModelRow, sha: str, family: str, offline: Optional[Mapping[str, Any]], live: Optional[Mapping[str, Any]]) -> dict:
    ov = (offline or {}).get("verdict") if offline and offline.get("status") == "OK" else None
    lv = (live or {}).get("verdict") if live and live.get("status") == "OK" else None
    verdict = P.combined_verdict(ov, lv)
    doc = {"family": family, "model": row.slug, "model_key": row.model_key, "sha": sha[:7], "parser": P.PARSER, "verdict": verdict,
           "offline": ({"status": (offline or {}).get("status"), "verdict": ov} if offline is not None else None),
           "live": ({"status": (live or {}).get("status"), "verdict": lv} if live is not None else None),
           "on_pass": P.on_pass_actions(family) if verdict == P.VERDICT_PASS else [],
           "at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    _write(_out_dir(P, row, sha) / "verdict.json", doc)
    return doc


@app.function(image=IMAGE, volumes={T.VOL_PATH: T.VOL}, cpu=4, memory=16_384, timeout=3600)
def check_tool_parser_offline(sha: str, family: str) -> dict:
    rec = _check_offline(sha, family)
    print("[tinycap] REC", json.dumps({k: v for k, v in rec.items() if k not in ("results", "either_of", "edges")}, default=str))
    return rec


@app.function(image=IMAGE, volumes={T.VOL_PATH: T.VOL}, gpu="H200", timeout=3 * 3600)
def probe_tool_parser(sha: str, family: str) -> dict:
    rec = _probe_live(sha, family)
    print("[tinycap] REC", json.dumps({k: v for k, v in rec.items() if k != "items"}, default=str))
    return rec


@app.function(image=IMAGE_NEMOTRON, volumes={T.VOL_PATH: T.VOL}, gpu="H200",
              cpu=T.NEMOTRON_FUNCTION_CPU, timeout=3 * 3600)
def probe_tool_parser_nemotron(sha: str, family: str) -> dict:
    rec = _probe_live(sha, family)
    print("[tinycap] REC", json.dumps({k: v for k, v in rec.items() if k != "items"}, default=str))
    return rec


def live_function(family: str):
    """The GPU function for a family: the Nemotron image (fused Mamba kernels, cpu=8) for
    nemotron_h — a Nemotron standup on the plain image serves UNATTESTED and is refused."""
    return probe_tool_parser_nemotron if family == "nemotron_h" else probe_tool_parser


@app.function(image=IMAGE, volumes={T.VOL_PATH: T.VOL}, cpu=2, timeout=4 * 3600)
def run_probe(sha: str, family: str, offline: bool = True, live: bool = True) -> dict:
    """Both halves in order (CPU first — a failing offline check spends no GPU), then the verdict.
    A missing staged tree is a NO_TREE verdict document, never a traceback (code review)."""
    try:
        tree_path, _ = T.resolve_staged_tree(sha)
    except (T.NoStagedTree, T.TreeShaMissing, T.TreeShaMismatch) as e:
        return {"family": family, "sha": sha[:7], "status": "NO_TREE", "verdict": "NO_TREE", "on_pass": [],
                "error": f"{type(e).__name__}: {e}", "offline": None, "live": None}
    P = _probe_module(str(tree_path))
    row = C.model_row(P.PROBE_ROWS[family])
    off = check_tool_parser_offline.remote(sha, family) if offline else None
    if off is not None and (off.get("status") != "OK" or off["verdict"]["verdict"] == P.VERDICT_FAIL):
        return _combine(P, row, sha, family, off, None)
    lv = live_function(family).remote(sha, family) if live else None
    return _combine(P, row, sha, family, off, lv)


@app.local_entrypoint()
def main(sha: str = "", family: str = "", offline_only: bool = False, live_only: bool = False, dry_run: bool = False):
    if not sha or not family:
        raise SystemExit("pass --sha <staged tree sha> --family qwen3_5|nemotron_h")
    sys.path.insert(0, str(REPO_LOCAL / "scripts"))
    import bfcl_parser_probe as PL
    if family not in PL.PROBE_ROWS:
        raise SystemExit(f"--family {family!r} not in {sorted(PL.PROBE_ROWS)}")
    row = C.model_row(PL.PROBE_ROWS[family])
    print(f"[parser-probe] {family} -> {row.slug} ({row.hf}@{row.rev[:12]}), parser {PL.PARSER}, manifest {PL.PROBE_MANIFESTS[family]}, "
          f"app {APP_NAME!r}; {PL.PROBE_N} items; image {'nemotron' if family == 'nemotron_h' else 'serving'}")
    if getattr(T, "IMAGE_NEMOTRON_ERROR", None) and family == "nemotron_h":
        raise SystemExit(f"[parser-probe] the Nemotron image is unbound on this client: {T.IMAGE_NEMOTRON_ERROR}")
    if dry_run:
        print("[parser-probe] dry run: nothing launched")
        return
    doc = run_probe.remote(sha, family, not live_only, not offline_only)
    print(json.dumps(doc, indent=1, default=str))
    print(f"[parser-probe] VERDICT {doc['verdict']} for {family} ({row.slug})")
    if doc["verdict"] == PL.VERDICT_PASS:
        print("[parser-probe] on PASS, land in ONE change:\n  - " + "\n  - ".join(doc["on_pass"]))
    else:
        raise SystemExit(f"[parser-probe] {doc['verdict']}: the {family} BFCL legs stay BLOCKED")
