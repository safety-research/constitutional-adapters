#!/usr/bin/env python3
"""The C/P/I track's MiniMax-M3 judge, LIVE slot: the never-submitted keys of the track's batch run directories (OR-Bench hard + XSTest
C/P/I ``cpi``, the XSTest contrast half ``option_xstest_contrast``, and the agentic classifier ``agentic``, which shares the code) judged
on OpenRouter chat completions with the batch track's exact request bodies. The engine is ``rejudge.common.live`` (its module docstring
is the contract); this module adds the track's parts: one live run directory per batch run directory, a routing chosen from what the
bodies carry, the track harvester's moderation-share hold, a guard on the batch submit gate, one process for all the runs, and their
handovers.

    cd <checkout>; export PYTHONPATH=$PWD/src:$PWD/scripts
    python scripts/rejudge/terra/live.py init --terra-dir T --runs cpi --max-spend-usd cpi=USD --global-root ROOT --global-ceiling-usd N
                                                            # the live run directories and their key sets (no network)
    python scripts/rejudge/terra/live.py init --refresh ...  # after the batch gate is closed: the key sets again (no live attempt yet)
    python scripts/rejudge/terra/live.py verify --terra-dir T --runs cpi        # the key sets against the sources, body by body (read only)
    python scripts/rejudge/terra/live.py dry-run --terra-dir T --runs cpi       # keys, cost bands, duration per run and in total (no network)
    python scripts/rejudge/terra/live.py run-all --terra-dir T --runs cpi [--limit N]   # the engines in the foreground ($OPENROUTER_API_KEY)
    python scripts/rejudge/terra/live.py run --terra-dir T --run cpi [--limit N]        # one engine
    python scripts/rejudge/terra/live.py status --terra-dir T                   # read only
    python scripts/rejudge/terra/live.py handover --terra-dir T --runs cpi --reasks [--ingest]   # re-asks a gated batch harvester will not build
    python scripts/rejudge/terra/live.py handover --terra-dir T --in-flight --reasks --not-before 2030-01-01T00:00:00Z --ingest
    python scripts/rejudge/terra/live.py reopen --terra-dir T --run cpi --provider NAME [--apply]
    python scripts/rejudge/terra/live.py settle-parked --terra-dir T --run cpi [--apply]   # a person's decision: leave the parked keys unjudged

THE TRACK DIRECTORY (``--terra-dir T``): ``T/full/<run>`` are the batch run directories (``build_manifests.py``), ``T/GO`` their submit
gate, and ``T/live/<run>`` (or ``--live-root``) the live run directories.

THE KEY SETS (``init``): per run, every key of ``T/full/<run>``'s PENDING batches (never POSTed: no batch id), read only from its ledger,
manifests and bodies (``rejudge.common.live.init_live``); never a key of a batch that is submitted, in flight or harvested. Every pending
body is read and must carry exactly the request parameters the judge entry builds for its family (``rejudge.terra.judges.protocol``),
which is what the routing is chosen from.

THE BATCH GATE. The batch run directories are built with ``--require-file T/GO``, so their harvester POSTs pending batches only while
``T/GO`` exists, and while it does the batch path still owns those keys: an ``init`` then marks each live run directory ``STAGED`` (a key
set for the dry run, not an owner), and ``run`` / ``run-all`` refuse a staged directory and refuse while the gate exists. To go live:
remove the GO file (the harvester keeps harvesting what is in flight and POSTs nothing more), wait for its next pass, then
``init --refresh`` (refused while the gate exists unless ``--stage`` keeps the result STAGED, while a source batch is mid-POST, or once a
live run directory has a live attempt or a handover) re-registers the key sets and clears ``STAGED``. ``init --refresh --stage``
re-registers a key set STAGED whatever the gate (a dry-run key set that owns nothing, e.g. after a code change); only a refresh without
it, with the gate closed, makes the live run the keys' owner. A running engine pauses within ``control_s`` if the gate file reappears,
and (the engine's own guard) when any batch it took leaves ``pending``. A run directory that exists only as a live run's source names a
gate file that is never created.

A PERSON'S EDITS SURVIVE A REFRESH. ``init`` writes ``<live dir>/track.defaults.json``, the settings as derived from this code, the source
and the flags; ``init --refresh`` compares the staged ``track.json`` with it and CARRIES OVER every edit (any ``live`` setting: the
concurrency, the routing, ...), printing each and recording them in ``init.json`` ``carried_over``; an explicit ``--concurrency`` wins
over both, and ``--discard-edits`` takes the defaults instead. A carried routing must allow only endpoints that declare every parameter
of the bodies (refused otherwise).

THE ROUTING (``track.json`` ``live.routing``; OpenRouter's provider-routing object). The fp8 endpoints of minimax/minimax-m3 declare
(https://openrouter.ai/api/v1/models/minimax/minimax-m3/endpoints, read 2026-11-11): ``tools`` and ``tool_choice`` (auto only,
``function: false``) everywhere, as the batch endpoint ``deepinfra/fp8/flex`` does; ``response_format`` on DeepInfra, GMICloud and
StreamLake but NOT on Parasail fp8; ``structured_outputs`` on none of them, nor on the batch endpoint. With ``require_parameters`` off (it
refuses every request: no fp8 endpoint supports the forced ``tool_choice``; see ``rejudge.common.live.default_routing``) OpenRouter would
drop a parameter an endpoint does not declare, silently. So a run's ``only`` is every fp8 endpoint that declares EVERY parameter its
bodies carry (:func:`routing_for`):

    cpi, option_xstest_contrast   messages + tools + tool_choice + reasoning + max_tokens: DeepInfra, GMICloud, StreamLake, Parasail
                                  (the StrongREJECT live routing, unchanged)
    agentic                       messages + response_format + reasoning + max_tokens: DeepInfra, GMICloud, StreamLake; Parasail fp8 is
                                  excluded (``ignore`` gets ``parasail/fp8`` as well) because it would drop the strict json_schema

``quantizations: ["fp8"]``, ``allow_fallbacks`` true, ``require_parameters`` false, ``max_price`` at the dearest allowed endpoint
(0.30 / 1.20 USD per MTok), ``ignore`` Parasail's ``fast`` tier, CoreWeave (fp4; honours a forced function, unlike the batch endpoint),
Venice and MiniMax's own endpoint (quantization unknown). Every endpoint left declares the body's parameters exactly as the batch
endpoint does, so a live verdict is produced the way the batch ones were: the forced tool choice degraded to auto, the json_schema
response format passed through. Every reply's serving provider is recorded, its quantization from this map. The routing and the
endpoints' slugs and quantizations are part of the judge's identity (``file_records.judge_spec``).

POLICY. The track's parsers (``rejudge.terra.cpi:parse_result``, ``rejudge.terra.agentic:parse_result``) and re-ask table (``track.json``
``retry``, counted within an outcome's class: the engine's terminal rule is the track harvester's) decide every outcome; a cut agentic
verdict is re-asked once at 8000. The track harvester's MODERATION-SHARE HOLD applies per recorded window (:class:`TerraLiveRunner`): a
window whose replies are at least ``health.moderation_share`` moderation (and at least ``health.min_count``) records them ``held`` and
pauses the run until a person reads them and writes ``moderation_share`` into ``<live dir>/ACKNOWLEDGED``.

REASONING. A reply that bills reasoning tokens is off-protocol (held, re-sent, never filed). ``reasoning_reroute`` is ON for these runs:
a key past ``max_reasoning_resends`` (3) such replies is re-sent with ``only`` narrowed to the allowed endpoints that have not billed
reasoning on it, and is PARKED (exit 5) only once every allowed endpoint has billed reasoning on it. A full run whose only unfinished keys
are parked ends ``unfinished`` (exit 5), and a restart re-sends each parked key once. ``settle-parked --apply`` (a person's decision,
engine stopped) makes the parked keys terminal unjudged so the filing can go ahead.

SPEND. Nothing has a default: ``init`` requires ``--max-spend-usd`` for every selected run (``run=USD,...``, or one USD with ``--run``;
USD incl. the 5.5% credit fee), ``--global-root`` (an absolute path of an existing directory: the engines' global check counts every
OpenRouter ledger under it, so put the track directory there) and ``--global-ceiling-usd`` (the ceiling of that check); a refresh restates
all three. The key is read from ``$OPENROUTER_API_KEY`` and never printed, logged or written.

ONE PROCESS (``run-all``): the engines as child processes (each its own writer lock, ledger, PAUSED file, signals and
``<live dir>/unit.log``), a combined ``<live root>/progress.json`` every 30 s, and the exit code combined like ``harvest_all.py``'s: 4
when any run paused, 5 when any is unfinished, 3 when any stopped, 0 when all are done (or a ``--limit`` pilot's keys are terminal); any
other child exit (a refusal, a crash) is returned as it is. ``--limit N`` splits N new keys over the runs (cpi 1/2, agentic 1/3, the
contrast half 1/6). A person-less restart after a handover checks every run's last engine run first
(``rejudge.common.live.auto_start_refusal`` with ``allow_parked``; ``status`` prints it).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import live as LV  # noqa: E402
from rejudge.common.state import Ledger, WriterLocked, readiness, utc_now  # noqa: E402
from rejudge.terra import harvest as TH  # noqa: E402
from rejudge.terra import judges as J  # noqa: E402

#: the track's batch run directories under ``<terra dir>/full/`` (``build_manifests.py --family cpi`` for ``cpi`` and the XSTest contrast
#: half ``option_xstest_contrast``, ``--family agentic`` for ``agentic``), one live run directory each
RUNS = ("cpi", "agentic", "option_xstest_contrast")
ENTRY = os.path.abspath(__file__)
JUDGE_ENTRY = "minimax-m3"
STAGED_FILE = "STAGED"
#: the settings ``init`` derived from the code and the source, beside ``track.json``, so a refresh can tell a person's edits from them
DEFAULTS_FILE = "track.defaults.json"
#: per run: the target concurrency (300 together, split by the expected work: keys x the latency their prompts and replies imply)
CONCURRENCY = {"cpi": 140, "agentic": 120, "option_xstest_contrast": 40}
#: how a ``--limit N`` pilot splits N over the runs
PILOT_SHARE = {"cpi": 1 / 2, "agentic": 1 / 3, "option_xstest_contrast": 1 / 6}
#: the dry run's assumed latency per run until it has replies of its own: the StrongREJECT live slot's measured mean (4.5 s for ~1,500
#: prompt + ~310 completion tokens on these endpoints), doubled for the agentic prompts (~3,100 + ~440 tokens)
ASSUME_LATENCY_S = {"cpi": 4.5, "agentic": 9.0, "option_xstest_contrast": 4.5}
EXIT_REFUSED = 2

#: the live (synchronous) fp8 endpoints of minimax/minimax-m3, by the provider name a reply carries: their endpoint slug, quantization,
#: list price (input, output, cache read; USD per MTok) and the request parameters they declare (``supported_parameters`` of
#: https://openrouter.ai/api/v1/models/minimax/minimax-m3/endpoints, read 2026-11-11; GMICloud listed a 20% and StreamLake a 50%
#: discount, which the reported cost reflects). ``tool_choice`` is auto-only on every one (``supports_tool_choice.function`` false).
ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "DeepInfra": {"slug": "deepinfra/fp8", "quantization": "fp8", "price": [0.28, 1.10, 0.056],
                  "parameters": ("reasoning", "include_reasoning", "max_tokens", "temperature", "top_p", "stop", "frequency_penalty",
                                 "presence_penalty", "repetition_penalty", "top_k", "seed", "min_p", "tools", "tool_choice", "logit_bias",
                                 "response_format")},
    "GMICloud": {"slug": "gmicloud/fp8", "quantization": "fp8", "price": [0.24, 0.96, 0.048],
                 "parameters": ("reasoning", "include_reasoning", "max_tokens", "temperature", "top_p", "seed", "response_format", "tools",
                                "tool_choice")},
    "StreamLake": {"slug": "streamlake/fp8", "quantization": "fp8", "price": [0.30, 1.20, 0.06],
                   "parameters": ("reasoning", "include_reasoning", "response_format", "max_tokens", "temperature", "top_p", "stop", "tools",
                                  "tool_choice")},
    "Parasail": {"slug": "parasail/fp8", "quantization": "fp8", "price": [0.30, 1.20, 0.06],
                 "parameters": ("reasoning", "include_reasoning", "max_tokens", "temperature", "top_p", "frequency_penalty",
                                "presence_penalty", "repetition_penalty", "seed", "stop", "top_k", "logit_bias", "tools", "tool_choice",
                                "logprobs", "top_logprobs")},
}
#: the batch endpoint every batch verdict came from (DeepInfra's flex tier): the same declaration class (tools, auto-only tool_choice,
#: response_format, no structured_outputs)
BATCH_ENDPOINT = {"slug": "deepinfra/fp8/flex", "quantization": "fp8",
                  "parameters": ("reasoning", "include_reasoning", "max_tokens", "temperature", "top_p", "stop", "frequency_penalty",
                                 "presence_penalty", "repetition_penalty", "top_k", "seed", "min_p", "logit_bias", "response_format", "tools",
                                 "tool_choice"),
                  "supports_tool_choice": {"none": True, "auto": True, "required": False, "function": False}}
IGNORE = ("parasail/fast", "coreweave", "venice", "minimax")
ENDPOINTS_SOURCE = "https://openrouter.ai/api/v1/models/minimax/minimax-m3/endpoints (read 2026-11-11)"


# =====================================================================================================================================
# paths
# =====================================================================================================================================
@dataclasses.dataclass
class Paths:
    """The track directory (``full/<run>`` the batch sources, ``GO`` the batch submit gate) and the live root (``<run>`` per run)."""
    terra_dir: str
    live_root: Optional[str] = None

    def __post_init__(self) -> None:
        self.terra_dir = os.path.abspath(os.path.expanduser(self.terra_dir))
        self.live_root = os.path.abspath(os.path.expanduser(self.live_root or os.path.join(self.terra_dir, "live")))

    @property
    def gate_file(self) -> str:
        return os.path.join(self.terra_dir, "GO")

    def source(self, run: str) -> str:
        return os.path.join(self.terra_dir, "full", run)

    def live(self, run: str) -> str:
        return os.path.join(self.live_root, run)

    def gate_open(self) -> bool:
        return os.path.exists(self.gate_file)


def staged(live_dir: str) -> Optional[str]:
    p = os.path.join(live_dir, STAGED_FILE)
    if os.path.exists(p):
        with open(p) as fh:
            return fh.read().strip() or "STAGED"
    return None


# =====================================================================================================================================
# per-run flags
# =====================================================================================================================================
def parse_per_run(value: Optional[str], runs: Sequence[str], cast=float, flag: str = "--max-spend-usd") -> Dict[str, Any]:
    """``--max-spend-usd`` / ``--concurrency`` of ``init``: ``run=VALUE,...``, or one VALUE when exactly one run is selected (a bare value
    over several runs would give each the same ceiling, which is never what is meant)."""
    if value is None or str(value).strip() == "":
        return {}
    v = str(value).strip()
    if "=" not in v:
        if len(runs) != 1:
            raise SystemExit(f"{flag} {v} names no run; give run=VALUE,... (e.g. cpi=20,option_xstest_contrast=5) or select one run with --run")
        return {runs[0]: cast(v)}
    out: Dict[str, Any] = {}
    for part in v.split(","):
        if not part.strip():
            continue
        r, _, x = part.partition("=")
        r = r.strip()
        if r not in runs:
            raise SystemExit(f"{flag}: {r!r} is not one of the selected runs {list(runs)}")
        out[r] = cast(x.strip())
    return out


# =====================================================================================================================================
# the routing: from the parameters the bodies carry
# =====================================================================================================================================
def request_parameters(body: Mapping[str, Any]) -> Tuple[str, ...]:
    """The request parameters a chat-completions body carries (every top-level key but the messages)."""
    return tuple(sorted(k for k in body if k != "messages"))


def expected_parameters(spec: Mapping[str, Any]) -> Tuple[str, ...]:
    """The parameters the judge entry builds for this run directory's family (``rejudge.terra.judges.protocol``'s body template, plus
    ``max_tokens``, which the protocol leaves out of the identity)."""
    tmpl = J.protocol(spec.get("judge") or JUDGE_ENTRY, spec["family"])["body_template"]
    return tuple(sorted(set(request_parameters(tmpl)) | {"max_tokens"}))


def routing_for(params: Sequence[str], endpoints: Mapping[str, Mapping[str, Any]] = ENDPOINTS,
                ignore: Sequence[str] = IGNORE) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """``(providers, routing, basis)`` for bodies carrying ``params``: every fp8 endpoint that declares EVERY one of them (OpenRouter
    drops an undeclared parameter silently when ``require_parameters`` is off), each excluded endpoint's slug added to ``ignore``, and
    the basis of the choice (the parameters, the endpoints excluded and why, the batch endpoint's declaration) for ``track.json``."""
    params = tuple(sorted(params))
    keep: Dict[str, Dict[str, Any]] = {}
    excluded: Dict[str, List[str]] = {}
    for name, e in endpoints.items():
        missing = sorted(set(params) - set(e["parameters"]))
        if missing:
            excluded[name] = missing
        else:
            keep[name] = {"slug": e["slug"], "quantization": e["quantization"], "price": list(e["price"])}
    if not keep:
        raise SystemExit(f"REFUSED: no fp8 endpoint declares every parameter the bodies carry {list(params)}")
    batch_missing = sorted(set(params) - set(BATCH_ENDPOINT["parameters"]))
    routing = LV.default_routing(keep, quantizations=("fp8",), ignore=list(ignore) + [endpoints[n]["slug"] for n in sorted(excluded)])
    basis = {"body_parameters": list(params), "endpoints_declaring_all": sorted(keep), "excluded": excluded,
             "batch_endpoint": BATCH_ENDPOINT["slug"], "batch_endpoint_missing": batch_missing,
             "rule": "only the fp8 endpoints that declare every request parameter the batch bodies carry (require_parameters is off, so "
                     "OpenRouter would drop an undeclared one silently); tool_choice is auto-only on every one, as on the batch endpoint",
             "reasoning_reroute": "a key whose replies billed reasoning more than max_reasoning_resends times is re-sent with only narrowed to "
                                  "the endpoints above that have not billed reasoning on it (never widened; the body unchanged), and parked "
                                  "once none is left",
             "endpoints_source": ENDPOINTS_SOURCE}
    return keep, routing, basis


def check_entry(model: str) -> None:
    e = J.get(JUDGE_ENTRY)
    if model != e["openrouter_model"] or LV.FORBIDDEN.search(model):
        raise SystemExit(f"REFUSED: the source run's model {model!r} is not the {JUDGE_ENTRY} entry's {e['openrouter_model']!r}")


def pending_parameters(src: str) -> collections.Counter:
    """``Counter(parameter tuple -> requests)`` over every request of the source's PENDING batches (read only; each body sha256-checked)."""
    con = sqlite3.connect(f"file:{os.path.join(src, 'state.sqlite')}?mode=ro", uri=True, timeout=60)
    try:
        bks = [r[0] for r in con.execute("SELECT batch_key FROM batches WHERE state IN ('pending', 'balance_queued') "
                                         "AND (batch_id IS NULL OR batch_id = '') ORDER BY batch_key")]
    finally:
        con.close()
    out: collections.Counter = collections.Counter()
    for bk in bks:
        man = BF.read_manifest(src, bk)
        for r in BF.body_requests(BF.read_body(src, man)):
            out[request_parameters(r["body"])] += 1
    return out


# =====================================================================================================================================
# init
# =====================================================================================================================================
def _source_mid_post(src: str) -> int:
    con = sqlite3.connect(f"file:{os.path.join(src, 'state.sqlite')}?mode=ro", uri=True, timeout=60)
    try:
        return int(con.execute("SELECT COUNT(*) FROM batches WHERE state = 'submit_unknown'").fetchone()[0])
    finally:
        con.close()


def _live_activity(live_dir: str) -> Dict[str, int]:
    path = os.path.join(live_dir, "state.sqlite")
    if not os.path.exists(path):
        return {"keys": 0, "attempts": 0, "handovers": 0, "runs": 0}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    try:
        def n(t: str) -> int:
            return int(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) if LV.has_table(con, t) else 0
        return {"keys": n("live_keys"), "attempts": n("live_attempts"), "handovers": n("live_handovers"),
                "runs": 1 if LV.meta_get(con, "run_state") else 0}
    finally:
        con.close()


def _supersede(live_dir: str, log) -> Optional[str]:
    """Move a staged ledger aside (never deleted) so the key set can be registered again; ``track.json`` and the defaults file are copied
    beside it (the refresh rewrites them)."""
    moved = [f for f in ("state.sqlite", "state.sqlite-wal", "state.sqlite-shm", "init.json", "dry_run.json", "verify.json", STAGED_FILE)
             if os.path.exists(os.path.join(live_dir, f))]
    if not moved:
        return None
    base = os.path.join(live_dir, "superseded", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    dest, j = base, 0
    while os.path.exists(dest):                           # two refreshes in one second
        j += 1
        dest = f"{base}-{j}"
    os.makedirs(dest, exist_ok=False)
    for f in ("track.json", DEFAULTS_FILE):
        if os.path.exists(os.path.join(live_dir, f)):
            shutil.copy2(os.path.join(live_dir, f), os.path.join(dest, f))
    for f in moved:
        os.replace(os.path.join(live_dir, f), os.path.join(dest, f))
    log(f"[init] {live_dir}: the staged key set moved to {dest} ({moved}; track.json copied)")
    return dest


_ABSENT = "<absent>"


def staged_edits(live_dir: str) -> Dict[str, Any]:
    """A person's edits to a staged ``track.json``: ``{key: value}`` for every top-level setting and ``{"live": {key: value}}`` for every
    ``live`` setting that differs from ``track.defaults.json`` (what ``init`` derived); a removed setting is ``"<absent>"``. Refused for
    a directory without the defaults file (an older init's), whose edits cannot be told from its defaults."""
    tp, dp = os.path.join(live_dir, "track.json"), os.path.join(live_dir, DEFAULTS_FILE)
    if not os.path.exists(tp):
        return {}
    if not os.path.exists(dp):
        raise SystemExit(f"REFUSED: {live_dir} has no {DEFAULTS_FILE} (an older init wrote it), so a person's edits to its track.json "
                         f"cannot be told from the defaults. Read its max_spend_usd and live settings; then refresh with --discard-edits, "
                         f"giving any setting you want kept as --max-spend-usd run=USD,... / --concurrency run=N,...")
    cur, dflt = BF.read_track(live_dir), json.load(open(dp))
    out: Dict[str, Any] = {}
    for k in sorted(set(cur) | set(dflt)):
        if k != "live" and cur.get(k, _ABSENT) != dflt.get(k, _ABSENT):
            out[k] = cur.get(k, _ABSENT)
    lc, lf = cur.get("live") or {}, dflt.get("live") or {}
    le = {k: lc.get(k, _ABSENT) for k in sorted(set(lc) | set(lf)) if lc.get(k, _ABSENT) != lf.get(k, _ABSENT)}
    if le:
        out["live"] = le
    return out


def _apply_edits(spec: Dict[str, Any], edits: Mapping[str, Any]) -> None:
    for k, v in edits.items():
        if k == "live":
            for lk, lv in v.items():
                if lv == _ABSENT:
                    spec["live"].pop(lk, None)
                else:
                    spec["live"][lk] = lv
        elif v == _ABSENT:
            spec.pop(k, None)
        else:
            spec[k] = v


def init_run(run: str, paths: Paths, max_spend_usd: float, global_ceiling_usd: float, global_root: str, refresh: bool = False,
             concurrency: Optional[int] = None, live_overrides: Optional[Mapping[str, Any]] = None, stage: bool = False,
             discard_edits: bool = False, log=print) -> Dict[str, Any]:
    """One run's live directory and key set (module docstring, THE KEY SETS / THE BATCH GATE / A PERSON'S EDITS SURVIVE A REFRESH). Read
    only on the source. ``max_spend_usd`` (this run's ceiling), ``global_root`` and ``global_ceiling_usd`` (the engine's global check) are
    required, at a refresh too. ``concurrency`` is a person's explicit setting: it wins over the default and over any carried edit."""
    if max_spend_usd is None or global_ceiling_usd is None or not global_root:
        raise SystemExit(f"REFUSED: {run}: init needs this run's --max-spend-usd, --global-root and --global-ceiling-usd (none has a default)")
    src, ld = paths.source(run), paths.live(run)
    st = BF.read_track(src)
    check_entry(str(st.get("model")))
    if st.get("judge") != JUDGE_ENTRY:
        raise SystemExit(f"REFUSED: {src} was built for judge {st.get('judge')!r}, not {JUDGE_ENTRY!r}")
    act = _live_activity(ld)
    superseded = None
    edits: Dict[str, Any] = {}
    discarded: Dict[str, Any] = {}
    if refresh:
        if paths.gate_open() and not stage:
            raise SystemExit(f"REFUSED: the batch submit gate {paths.gate_file} exists, so the batch harvester may still POST the pending "
                             f"batches; remove it, wait for the harvester's next pass, then refresh (--stage re-registers the key set "
                             f"STAGED while the gate is open, e.g. after a code change)")
        if act["attempts"] or act["handovers"] or act["runs"]:
            raise SystemExit(f"REFUSED: {ld} has already run ({act}); its key set is fixed (add keys with a handover)")
        n_mid = _source_mid_post(src)
        if n_mid:
            raise SystemExit(f"REFUSED: {src} has {n_mid} batch(es) mid-POST (submit_unknown); wait for the batch harvester to reconcile them")
        if discard_edits:
            try:
                discarded = staged_edits(ld)
            except SystemExit:
                discarded = {"unknown": f"no {DEFAULTS_FILE}: the old track.json is kept in the superseded directory"}
        else:
            edits = staged_edits(ld)
        edits.pop("max_spend_usd", None)                  # the ceiling is restated at every init, never carried
        discarded.pop("max_spend_usd", None)
    elif act["keys"]:
        raise SystemExit(f"REFUSED: {ld} already holds {act['keys']} keys (init runs once; init --refresh re-registers a staged key set)")
    want = expected_parameters(st)
    got = pending_parameters(src)
    if set(got) != {want}:
        raise SystemExit(f"REFUSED: {src}'s pending bodies carry the parameters {dict(got)}, not exactly the {JUDGE_ENTRY} entry's "
                         f"{list(want)}; the routing is chosen from them")
    providers, routing, basis = routing_for(want)
    live = {"concurrency": int(CONCURRENCY[run]), "global_ceiling_usd": float(global_ceiling_usd), "routing_basis": basis,
            "reasoning_reroute": True, "global_root": LV.check_global_root(global_root)}
    live.update(live_overrides or {})
    ce = edits.get("live") or {}
    if "routing" in ce or "providers" in ce:
        r2 = routing if ce.get("routing", _ABSENT) == _ABSENT else ce["routing"]
        p2 = providers if ce.get("providers", _ABSENT) == _ABSENT else ce["providers"]
        extra = sorted(LV.allowed_providers(p2, r2) - set(providers))
        if extra or not LV.allowed_providers(p2, r2):
            raise SystemExit(f"REFUSED: the edited routing of {ld}/track.json allows {extra or 'no endpoint'}, which do(es) not declare every "
                             f"parameter the bodies carry ({list(want)}; allowed: {sorted(providers)}); fix it, or --discard-edits")
    if refresh:
        superseded = _supersede(ld, log)
    info = LV.init_live(ld, src, routing, providers, float(max_spend_usd), live=live, log=log)
    spec = BF.read_track(ld)
    spec.update({"family": st["family"], "terra_run": run, "batch_gate_file": paths.gate_file, "scope": st.get("scope"),
                 "keys_file": st.get("keys_file"), "exclude_keys_file": st.get("exclude_keys_file"), "terra_dir": paths.terra_dir,
                 "live_root": paths.live_root})
    with open(os.path.join(ld, DEFAULTS_FILE), "w") as fh:
        json.dump(spec, fh, indent=1)
    _apply_edits(spec, edits)
    explicit: Dict[str, Any] = {"max_spend_usd": float(max_spend_usd)}
    if concurrency is not None:
        spec["live"]["concurrency"] = int(concurrency)
        explicit["live.concurrency"] = int(concurrency)
    BF.write_track(ld, spec)
    for k, v in edits.items():
        for kk, vv in (v.items() if k == "live" else [(None, v)]):
            name = f"live.{kk}" if kk else k
            log(f"[init] {run}: carried over the edit {name} = {json.dumps(vv)[:200]}" + (" (then overridden by the flag)" if name in explicit else ""))
    if discarded:
        log(f"[init] {run}: --discard-edits: took the defaults instead of {json.dumps(discarded)[:400]}")
    gate = paths.gate_open()
    is_staged = gate or stage
    sp = os.path.join(ld, STAGED_FILE)
    if gate:
        with open(sp, "w") as fh:
            fh.write(f"{utc_now()} staged: the batch submit gate {paths.gate_file} existed at init, so the batch harvester still owns "
                     f"these keys and may POST their batches. This key set is for the dry run. To go live: remove the gate file, wait for "
                     f"the harvester's next pass, then `terra/live.py init --refresh` (which replaces this file's key set and removes it).\n")
    elif stage:
        with open(sp, "w") as fh:
            fh.write(f"{utc_now()} staged: registered with --stage (the batch gate was closed): a key set for the verify and the dry run that "
                     f"owns nothing. To go live: `terra/live.py init --refresh` (which replaces this file's key set and removes it).\n")
    elif os.path.exists(sp):
        os.remove(sp)
    out = {"utc": utc_now(), "run": run, "live_dir": ld, "source_run_dir": src, "staged": is_staged, "superseded": superseded,
           "body_parameters": list(want), "requests_by_parameters": {",".join(k): v for k, v in got.items()},
           "routing": spec["live"]["routing"], "providers": sorted(providers), "max_spend_usd": spec["max_spend_usd"],
           "concurrency": spec["live"]["concurrency"], "carried_over": edits, "explicit": explicit, "discarded_edits": discarded,
           **{k: v for k, v in info.items() if k not in ("calibration", "utc")}}
    with open(os.path.join(ld, "init.json"), "w") as fh:
        json.dump(dict(out, calibration=info.get("calibration")), fh, indent=1)
    return out


# =====================================================================================================================================
# verify: the key sets against the sources (read only)
# =====================================================================================================================================
def verify_run(run: str, paths: Paths, bodies: bool = True) -> Dict[str, Any]:
    """The live key set == the source's non-terminal keys of its PENDING (never POSTed) batches; no overlap with any other batch state,
    with the reuse ledgers the source excluded, or between runs (checked by the caller); every key in the scope's unit map; with
    ``bodies``, every registered body sha256 equal to the source manifest body's (read and hash-checked)."""
    src, ld = paths.source(run), paths.live(run)
    lcon = sqlite3.connect(f"file:{os.path.join(ld, 'state.sqlite')}?mode=ro", uri=True, timeout=60)
    try:
        live = {r[0]: (r[1], r[2]) for r in lcon.execute("SELECT custom_id, source_batch_key, body_sha256 FROM live_keys")}
        origins = dict(lcon.execute("SELECT origin, COUNT(*) FROM live_keys GROUP BY 1").fetchall())
    finally:
        lcon.close()
    con = sqlite3.connect(f"file:{os.path.join(src, 'state.sqlite')}?mode=ro", uri=True, timeout=60)
    try:
        states = {r[0]: (r[1], r[2]) for r in con.execute("SELECT batch_key, state, batch_id FROM batches")}
        by_state: Dict[str, set] = collections.defaultdict(set)
        for cid, bk, term in con.execute("SELECT custom_id, batch_key, terminal FROM requests"):
            s, bid = states.get(bk, ("?", None))
            pend = s in LV.SOURCE_PENDING_STATES and not bid
            by_state["pending_nonterminal" if pend and not term else ("pending_terminal" if pend else s)].add(cid)
    finally:
        con.close()
    spec = BF.read_track(src)
    lk = set(live)
    out: Dict[str, Any] = {"run": run, "live_keys": len(lk), "origins": origins, "source_pending_nonterminal": len(by_state["pending_nonterminal"]),
                           "equal_to_source_pending": lk == by_state["pending_nonterminal"] if origins.keys() <= {"pending"} else None,
                           "overlap_by_source_state": {s: len(lk & ks) for s, ks in sorted(by_state.items()) if s != "pending_nonterminal" and lk & ks}}
    excl = spec.get("exclude_keys_file")
    if excl and os.path.exists(excl):
        ek = {l.strip() for l in open(excl) if l.strip()}
        out["overlap_with_excluded_reuse_keys"] = len(lk & ek)
    scope = spec.get("scope")
    if scope and os.path.exists(os.path.join(scope, "units.tsv.gz")):
        import gzip
        with gzip.open(os.path.join(scope, "units.tsv.gz"), "rt") as fh:
            hdr = next(fh).rstrip("\n").split("\t")
            ci = hdr.index("custom_id")
            uk = {line.rstrip("\n").split("\t")[ci] for line in fh}
        out["keys_not_in_scope_units"] = len(lk - uk)
    if bodies:
        bad = 0
        by_batch: Dict[str, List[str]] = collections.defaultdict(list)
        for cid, (bk, _sha) in live.items():
            by_batch[bk].append(cid)
        for bk, cids in sorted(by_batch.items()):
            got = {r["custom_id"]: LV.body_sha256(r["body"]) for r in BF.body_requests(BF.read_body(src, BF.read_manifest(src, bk)))}
            bad += sum(1 for c in cids if got.get(c) != live[c][1])
        out["body_sha256_mismatches"] = bad
        out["source_batches_read"] = len(by_batch)
    out["ok"] = (out.get("equal_to_source_pending") in (True, None) and not out["overlap_by_source_state"]
                 and not out.get("overlap_with_excluded_reuse_keys") and not out.get("keys_not_in_scope_units")
                 and not out.get("body_sha256_mismatches"))
    return out


# =====================================================================================================================================
# the engine: the shared one plus the track harvester's moderation-share hold and the batch-gate guard
# =====================================================================================================================================
class TerraLiveRunner(LV.LiveRunner):
    """:class:`rejudge.common.live.LiveRunner` with two of the track's rules: the moderation-share hold of :mod:`rejudge.terra.harvest`,
    per recorded window, and a pause when the batch submit gate reappears (the batch harvester could then POST batches whose keys this
    run owns)."""

    def __init__(self, live_dir: str, gate_file: Optional[str] = None, **kw: Any):
        super().__init__(live_dir, **kw)
        self.gate_file = gate_file or self.spec.get("batch_gate_file")
        self.health_cfg = {"moderation_share": TH.MODERATION_SHARE, "min_count": TH.MIN_COUNT, **self.health_cfg}

    def _moderation_hold(self, window: Sequence[LV.Result]) -> Optional[str]:
        """Records a window's moderation replies ``held`` (and returns the pause reason) when they are at least ``moderation_share`` of it
        and at least ``min_count``, unless ``moderation_share`` is acknowledged (``ACKNOWLEDGED`` or ``health.acknowledged``)."""
        cfg = self.health_cfg
        if TH.ACK_MODERATION in TH.acknowledged(self.live_dir, cfg.get("acknowledged")):
            return None
        n = len(window)
        mod = [r for r in window if r.parsed.get("outcome") == "moderation"]
        if not n or len(mod) < int(cfg["min_count"]) or len(mod) / n < float(cfg["moderation_share"]):
            return None
        for r in mod:
            p = r.parsed
            r.parsed = {"outcome": "held", "usage": p.get("usage"), "status_code": p.get("status_code"), "finish_reason": p.get("finish_reason"),
                        "error": {"kind": "held_moderation_share", "parsed_outcome": "moderation", "parsed_error": p.get("error")},
                        "sent_sha256": p.get("sent_sha256")}
        return (f"moderation share {len(mod)} of {n} replies ({len(mod) / n:.1%}) >= {float(cfg['moderation_share']):.1%}: HELD, not recorded "
                f"as moderation. Read the rejections (the attempts' error); if they are real, add the line {TH.ACK_MODERATION} to "
                f"{os.path.join(self.live_dir, TH.ACK_FILE)} and remove PAUSED (the held keys are re-sent and recorded as what they are); "
                f"otherwise fix the cause and remove PAUSED")

    def _health(self, window: Sequence[LV.Result]):
        why = self._moderation_hold(window)               # before the shared rules, which then see those replies as held
        held, reasons, cool = super()._health(window)
        if why:
            reasons = list(reasons) + [why]
            cool = False
        return held, reasons, cool

    def _controls(self) -> None:
        super()._controls()
        if self.gate_file and os.path.exists(self.gate_file) and not self.stopping:
            self.pause(f"the batch submit gate {self.gate_file} exists: the batch harvester may POST the pending batches whose keys "
                       f"this live run owns. Remove the gate file (the live run owns these keys), then remove PAUSED and start again")


def run_one(run: str, paths: Paths, limit: Optional[int] = None, overrides: Optional[Mapping[str, Any]] = None,
            max_spend_usd: Optional[float] = None, log=print) -> int:
    ld = paths.live(run)
    why = staged(ld)
    if why:
        raise SystemExit(f"REFUSED: {ld} is STAGED ({why.splitlines()[0][:300]})")
    if paths.gate_open():
        raise SystemExit(f"REFUSED: the batch submit gate {paths.gate_file} exists, so the batch path still owns the pending keys; "
                         f"remove it, wait for the harvester's next pass, then `init --refresh`")
    r = TerraLiveRunner(ld, gate_file=paths.gate_file, overrides=overrides, limit=limit, max_spend_usd=max_spend_usd, log=log)
    check_entry(r.model)
    return asyncio.run(r.run_async())


# =====================================================================================================================================
# all the runs in one process
# =====================================================================================================================================
PRIORITY = {LV.EXIT_PAUSED: 3, LV.EXIT_UNFINISHED: 2, LV.EXIT_STOPPED: 1, LV.EXIT_DONE: 0}


def combined_code(codes: Sequence[int]) -> int:
    """A refusal or crash (any code the engine does not use) is returned as it is; else PAUSED over UNFINISHED over STOPPED over DONE."""
    odd = [c for c in codes if c not in PRIORITY]
    if odd:
        return odd[0]
    return max(codes, key=lambda c: PRIORITY[c]) if codes else LV.EXIT_DONE


def split_limit(limit: int, runs: Sequence[str]) -> Dict[str, int]:
    """``limit`` new keys over ``runs`` by :data:`PILOT_SHARE` (each at least 1; the shares renormalized over the runs given)."""
    tot = sum(PILOT_SHARE[r] for r in runs)
    out = {r: max(1, int(round(limit * PILOT_SHARE[r] / tot))) for r in runs}
    diff = int(limit) - sum(out.values())
    out[runs[0]] = max(1, out[runs[0]] + diff)
    return out


def _read(path: str) -> Dict[str, Any]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def write_combined(paths: Paths, runs: Sequence[str], state: str, codes: Optional[Mapping[str, Optional[int]]] = None) -> Dict[str, Any]:
    """``<live root>/progress.json``: each run's state, keys, spend and ETA from its own ``progress.json``, and the totals."""
    per: Dict[str, Any] = {}
    tot: collections.Counter = collections.Counter()
    spend = 0.0
    rate = 0.0
    etas = []
    outcomes: collections.Counter = collections.Counter()
    for run in runs:
        p = _read(os.path.join(paths.live(run), "progress.json"))
        k = p.get("keys") or {}
        sp = p.get("spend_usd") or {}
        per[run] = {"state": p.get("state"), "updated_utc": p.get("updated_utc"), "exit_code": (codes or {}).get(run), "paused": p.get("paused"),
                    "keys": k, "outcomes": p.get("outcomes"), "terminal_keys_per_min": p.get("terminal_keys_per_min"),
                    "latency_s_last_5m": p.get("latency_s_last_5m"), "concurrency": {x: (p.get("concurrency") or {}).get(x) for x in ("limit", "target", "active", "halvings")},
                    "spend_usd": {x: sp.get(x) for x in ("spent_with_fee", "in_flight_with_fee", "max_spend_usd", "per_terminal_key_with_fee")},
                    "global": sp.get("global"), "off_protocol_held": p.get("off_protocol_held"), "parked_keys": p.get("parked_keys"),
                    "provider_mix": {n: {x: v.get(x) for x in ("replies", "share", "allowed", "mean_latency_s", "reasoning_token_replies")}
                                     for n, v in (p.get("provider_mix") or {}).items()},
                    "eta": p.get("eta")}
        for x in ("total", "terminal", "remaining", "sent"):
            tot[x] += int(k.get(x) or 0)
        outcomes.update({o: int(n) for o, n in (p.get("outcomes") or {}).items()})
        spend += float(sp.get("spent_with_fee") or 0.0)
        rate += float((p.get("terminal_keys_per_min") or {}).get("last_5m") or 0.0)
        if (p.get("eta") or {}).get("utc"):
            etas.append(p["eta"]["utc"])
    out = {"updated_utc": utc_now(), "state": state, "live_root": paths.live_root, "runs": per, "keys": dict(tot), "outcomes": dict(outcomes),
           "spent_with_fee_usd": round(spend, 4), "max_spend_usd": {r: (per[r]["spend_usd"] or {}).get("max_spend_usd") for r in runs},
           "terminal_keys_per_min_last_5m": round(rate, 1), "eta_utc_latest": max(etas) if etas else None,
           "batch_gate_open": paths.gate_open()}
    path = os.path.join(paths.live_root, "progress.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)
    return out


def run_all(paths: Paths, runs: Sequence[str] = RUNS, limit: Optional[int] = None, max_hours: Optional[float] = None,
            extra: Sequence[str] = (), progress_s: float = 30.0, python: str = sys.executable, log=print) -> int:
    """The engines as child processes (``live.py run --run R``), each logging to its own ``<live dir>/unit.log``; the combined progress
    file every ``progress_s``; the combined exit code. A SIGTERM / SIGINT is passed to the children, which finish their requests in flight
    and exit 3."""
    for run in runs:
        why = staged(paths.live(run))
        if why:
            log(f"REFUSED: {paths.live(run)} is STAGED ({why.splitlines()[0][:300]})")
            return EXIT_REFUSED
    if paths.gate_open():
        log(f"REFUSED: the batch submit gate {paths.gate_file} exists (the batch path still owns the pending keys)")
        return EXIT_REFUSED
    limits = split_limit(int(limit), runs) if limit else {}
    procs: Dict[str, subprocess.Popen] = {}
    logs = []
    forwarded: Dict[str, Any] = {"sig": None}

    def forward(signum, _frame):
        forwarded["sig"] = signum
        for p in procs.values():
            if p.poll() is None:
                try:
                    p.send_signal(signum)
                except OSError:
                    pass

    old = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
    for run in runs:
        if forwarded["sig"] is not None:
            log(f"{utc_now()} [run-all] signal {forwarded['sig']} before {run} started: not started")
            break
        argv = [python, ENTRY, "run", "--run", run, "--terra-dir", paths.terra_dir, "--live-root", paths.live_root]
        if max_hours is not None:
            argv += ["--max-hours", f"{max_hours:g}"]
        if run in limits:
            argv += ["--limit", str(limits[run])]
        argv += list(extra)
        fh = open(os.path.join(paths.live(run), "unit.log"), "a")
        fh.write(f"{utc_now()} [run-all] {' '.join(shlex.quote(a) for a in argv[1:])}\n")
        fh.flush()
        logs.append(fh)
        procs[run] = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT)
        log(f"{utc_now()} [run-all] started {run} (pid {procs[run].pid}{', limit ' + str(limits[run]) if run in limits else ''})")
    codes: Dict[str, Optional[int]] = {r: (None if r in procs else LV.EXIT_STOPPED) for r in runs}
    try:
        t_prog = 0.0
        while True:
            for run, p in procs.items():
                if codes[run] is None and p.poll() is not None:
                    codes[run] = p.returncode
                    log(f"{utc_now()} [run-all] {run} exited {p.returncode}")
            done = all(c is not None for c in codes.values())
            if done or time.monotonic() - t_prog >= progress_s:
                t_prog = time.monotonic()
                write_combined(paths, runs, "done" if done else ("stopping" if forwarded["sig"] else "running"), codes)
            if done:
                break
            time.sleep(min(1.0, progress_s))
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        for fh in logs:
            fh.close()
    code = combined_code([int(c) for c in codes.values() if c is not None])
    write_combined(paths, runs, {0: "done", 3: "stopped", 4: "paused", 5: "unfinished"}.get(code, f"exit {code}"), codes)
    log(f"{utc_now()} [run-all] exit {code}: {codes}")
    return code


# =====================================================================================================================================
# dry run (no network)
# =====================================================================================================================================
def dry_run_all(paths: Paths, runs: Sequence[str] = RUNS, total_concurrency: int = 300, global_root: Optional[str] = None) -> Dict[str, Any]:
    per: Dict[str, Any] = {}
    for run in runs:
        ld = paths.live(run)
        d = LV.dry_run(ld, assume_latency_s=ASSUME_LATENCY_S[run], latency_band=(3.0, 20.0), global_root=None)
        d["staged"] = bool(staged(ld))
        with open(os.path.join(ld, "dry_run.json"), "w") as fh:
            json.dump(d, fh, indent=1)
        per[run] = d
    bands: collections.Counter = collections.Counter()
    for d in per.values():
        for k, v in d["cost_usd_with_fee"].items():
            if k in ("low", "central", "high"):
                bands[k] += float(v)
    work = {r: per[r]["duration"]["requests_incl_reasks"] * per[r]["duration"]["latency_s"] for r in runs}
    hours_parallel = max((per[r]["duration"]["hours_at_latency"] for r in runs), default=0.0)
    out = {"utc": utc_now(), "live_root": paths.live_root, "batch_gate_open": paths.gate_open(),
           "keys": {r: per[r]["keys"]["remaining"] for r in runs}, "keys_total": sum(per[r]["keys"]["remaining"] for r in runs),
           "requests_incl_reasks": {r: per[r]["duration"]["requests_incl_reasks"] for r in runs},
           "cost_usd_with_fee": {r: per[r]["cost_usd_with_fee"] for r in runs}, "cost_usd_with_fee_total": {k: round(v, 2) for k, v in bands.items()},
           "max_spend_usd": {r: per[r]["max_spend_usd"] for r in runs},
           "concurrency": {r: per[r]["concurrency"] for r in runs},
           "duration": {"per_run_hours_at_its_concurrency": {r: per[r]["duration"]["hours_at_latency"] for r in runs},
                        "latency_s": {r: per[r]["duration"]["latency_s"] for r in runs},
                        "latency_source": {r: per[r]["duration"]["latency_source"] for r in runs},
                        "hours_all_three_in_parallel": hours_parallel,
                        f"hours_if_{total_concurrency}_shared_by_work": round(sum(work.values()) / total_concurrency / 3600.0, 2),
                        "hours_band_per_run": {r: per[r]["duration"]["hours_band"] for r in runs}},
           "tokens": {r: per[r]["tokens"] for r in runs}, "prices_per_mtok": {r: per[r]["prices_per_mtok"] for r in runs},
           "routing": {r: per[r]["routing"] for r in runs}}
    if global_root is not None:
        other = LV.openrouter_spend(global_root, exclude=[paths.live(r) for r in runs])
        ceil = min(float((BF.read_track(paths.live(r)).get("live") or {}).get("global_ceiling_usd") or 0.0) for r in runs) if runs else 0.0
        out["global"] = {"root": global_root, "other_ledgers_usd": other["usd"], "ceiling_usd": ceil,
                         "headroom_for_these_live_runs_usd": round(ceil - float(other["usd"]), 2),
                         "top_dirs": dict(sorted(other["per_dir"].items(), key=lambda kv: -kv[1])[:10]), "errors": other["errors"]}
    with open(os.path.join(paths.live_root, "dry_run.json"), "w") as fh:
        json.dump({"summary": out, "runs": per}, fh, indent=1)
    return out


# =====================================================================================================================================
# the CLI
# =====================================================================================================================================
def _runs(a: argparse.Namespace) -> List[str]:
    if getattr(a, "run", None):
        return [a.run]
    rs = [r.strip() for r in (getattr(a, "runs", None) or ",".join(RUNS)).split(",") if r.strip()]
    bad = [r for r in rs if r not in RUNS]
    if bad:
        raise SystemExit(f"unknown run(s) {bad}; one of {RUNS}")
    return rs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, run_choice: bool = True) -> None:
        p.add_argument("--terra-dir", required=True, help="the track directory: full/<run> (the batch sources) and GO (the batch gate)")
        p.add_argument("--live-root", default=None, help="default <terra dir>/live")
        if run_choice:
            p.add_argument("--run", choices=RUNS, help="one run (default: every run of --runs)")
            p.add_argument("--runs", default=",".join(RUNS), help=f"comma-separated runs (default {','.join(RUNS)})")

    pi = sub.add_parser("init", help="the live run directories and their key sets (no network)")
    common(pi)
    pi.add_argument("--refresh", action="store_true", help="re-register a STAGED key set (refused while the batch gate exists or after a "
                    "run); a person's edits to track.json are carried over")
    pi.add_argument("--stage", action="store_true", help="with --refresh: register the key set STAGED (owning nothing) whatever the batch "
                    "gate; allowed while the gate exists")
    pi.add_argument("--discard-edits", action="store_true", help="with --refresh: take the defaults instead of the staged track.json's edits")
    pi.add_argument("--max-spend-usd", required=True, help="every selected run's ceiling, run=USD,... (or one USD with --run); USD incl. "
                    "the credit fee")
    pi.add_argument("--concurrency", help="run=N,... (or one N with --run); default cpi 140, agentic 120, option_xstest_contrast 40")
    pi.add_argument("--global-ceiling-usd", type=float, required=True, help="the ceiling of the engines' global check")
    pi.add_argument("--global-root", required=True, help="the root the global check walks: an absolute path of an existing directory "
                    "holding every OpenRouter run directory whose spend counts against --global-ceiling-usd")
    pv = sub.add_parser("verify", help="the key sets against the sources, body by body (read only)")
    common(pv)
    pv.add_argument("--no-bodies", action="store_true")
    pd = sub.add_parser("dry-run", help="keys, cost, duration per run and in total (no network)")
    common(pd)
    pd.add_argument("--no-global", action="store_true", help="skip reading the other ledgers' spend")
    pr = sub.add_parser("run", help="one engine (foreground); needs $OPENROUTER_API_KEY")
    common(pr)
    pr.add_argument("--limit", type=int, help="pilot: at most N new keys (spread over the batches); exit 0 when they are terminal")
    pr.add_argument("--concurrency", type=int)
    pr.add_argument("--max-hours", type=float)
    pr.add_argument("--max-spend-usd", type=float)
    pr.add_argument("--global-check-s", type=float)
    pr.add_argument("--base-url", help="tests only: a local mock of OpenRouter")
    pa = sub.add_parser("run-all", help="every run's engine as a child process (foreground); needs $OPENROUTER_API_KEY")
    common(pa)
    pa.add_argument("--limit", type=int, help="pilot: N new keys split over the runs (cpi 1/2, agentic 1/3, contrast 1/6)")
    pa.add_argument("--max-hours", type=float)
    pa.add_argument("--progress-s", type=float, default=30.0)
    pa.add_argument("--base-url", help="tests only: a local mock of OpenRouter")
    pa.add_argument("--global-check-s", type=float, help="tests only")
    ph = sub.add_parser("handover", help="move keys the batch runs will not finish into the live runs (no network)")
    common(ph)
    ph.add_argument("--all", action="store_true", help="every run (the default when --run is not given)")
    ph.add_argument("--in-flight", action="store_true", help="keys of POSTed, unharvested batches (the fallback)")
    ph.add_argument("--reasks", action="store_true", help="keys waiting on a re-ask the gated batch harvester will never build")
    ph.add_argument("--not-before", help="refuse before this UTC time (YYYY-MM-DDTHH:MM:SSZ); for a scheduled fallback")
    ph.add_argument("--reason", default="")
    ph.add_argument("--ingest", action="store_true", help="register the keys now where no engine holds the live run's writer lock")
    po = sub.add_parser("reopen", help="make terminal live outcomes non-terminal again, so the engine judges those keys again")
    common(po)
    po.add_argument("--provider", action="append", default=[])
    po.add_argument("--off-protocol", action="store_true")
    po.add_argument("--reason", default="")
    po.add_argument("--apply", action="store_true")
    pk = sub.add_parser("settle-parked", help="a person's decision: make the PARKED keys (reasoning billed on every try) terminal unjudged")
    common(pk)
    pk.add_argument("--reason", default="")
    pk.add_argument("--apply", action="store_true", help="write (the engine must be stopped); without it, count only")
    ps = sub.add_parser("status", help="read only")
    common(ps)
    a = ap.parse_args(argv)
    paths = Paths(a.terra_dir, a.live_root)

    if a.cmd == "init":
        if (a.stage or a.discard_edits) and not a.refresh:
            raise SystemExit("--stage and --discard-edits go with --refresh")
        runs = _runs(a)
        root = LV.check_global_root(a.global_root)
        spend = parse_per_run(a.max_spend_usd, runs, float, "--max-spend-usd")
        unstated = [r for r in runs if r not in spend]
        if unstated:
            raise SystemExit(f"REFUSED: --max-spend-usd states no ceiling for {unstated} (run=USD,... for every selected run)")
        conc = parse_per_run(a.concurrency, runs, int, "--concurrency")
        os.makedirs(paths.live_root, exist_ok=True)
        os.chmod(paths.live_root, 0o700)
        out = {}
        for run in runs:
            out[run] = init_run(run, paths, max_spend_usd=spend[run], global_ceiling_usd=a.global_ceiling_usd, global_root=root,
                                refresh=a.refresh, concurrency=conc.get(run), stage=a.stage, discard_edits=a.discard_edits)
        print(json.dumps(out, indent=1, default=str))
        return 0
    if a.cmd == "verify":
        out = {run: verify_run(run, paths, bodies=not a.no_bodies) for run in _runs(a)}
        keysets = {}
        for run in _runs(a):
            con = sqlite3.connect(f"file:{os.path.join(paths.live(run), 'state.sqlite')}?mode=ro", uri=True)
            keysets[run] = {r[0] for r in con.execute("SELECT custom_id FROM live_keys")}
            con.close()
        rs = list(keysets)
        cross = {f"{x}&{y}": len(keysets[x] & keysets[y]) for i, x in enumerate(rs) for y in rs[i + 1:] if keysets[x] & keysets[y]}
        res = {"utc": utc_now(), "runs": out, "overlap_between_runs": cross, "ok": all(v["ok"] for v in out.values()) and not cross}
        with open(os.path.join(paths.live_root, "verify.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if a.cmd == "dry-run":
        rs = _runs(a)
        out = dry_run_all(paths, rs, global_root=None if a.no_global else LV.global_root_of(paths.live(rs[0])))
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "run":
        if not a.run:
            raise SystemExit("run needs --run (or use run-all)")
        ov = {"concurrency": a.concurrency, "max_hours": a.max_hours, "global_check_s": a.global_check_s, "base_url": a.base_url}
        return run_one(a.run, paths, limit=a.limit, overrides=ov, max_spend_usd=a.max_spend_usd)
    if a.cmd == "run-all":
        extra = (["--base-url", a.base_url] if a.base_url else []) + (["--global-check-s", f"{a.global_check_s:g}"] if a.global_check_s else [])
        return run_all(paths, _runs(a), limit=a.limit, max_hours=a.max_hours, extra=extra, progress_s=a.progress_s)
    if a.cmd == "handover":
        modes = (["in_flight"] if a.in_flight else []) + (["reasks"] if a.reasks else [])
        if not modes:
            raise SystemExit("name --in-flight and / or --reasks")
        reason = a.reason or ("fallback: the in-flight batches did not come back in time" if a.in_flight else
                              "re-asks the gated batch harvester will not build")
        out: Dict[str, Any] = {}
        for run in _runs(a):
            ld = paths.live(run)
            if staged(ld):
                out[run] = {"handover": None, "refused": "STAGED (init --refresh first)"}
                continue
            path = LV.build_handover(ld, modes, reason, not_before=a.not_before)
            added = 0
            if path and a.ingest:
                try:
                    led = Ledger(ld).writer()
                except WriterLocked:
                    print(f"{run}: the live engine holds the writer lock: it ingests the handover at its next check (within global_check_s)")
                else:
                    try:
                        LV.ensure_schema(led.conn)
                        added = LV.ingest_handovers(led, ld)
                    finally:
                        led.close()
            out[run] = {"handover": path, "ingested_now": added}
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "reopen":
        if not a.run:
            raise SystemExit("reopen needs --run")
        print(json.dumps(LV.reopen_live(paths.live(a.run), providers=a.provider, off_protocol=a.off_protocol, reason=a.reason, apply=a.apply),
                         indent=1))
        return 0
    if a.cmd == "settle-parked":
        out = {run: LV.settle_parked(paths.live(run), reason=a.reason, apply=a.apply) for run in _runs(a)
               if os.path.exists(os.path.join(paths.live(run), "state.sqlite"))}
        print(json.dumps(out, indent=1, default=str))
        return 0
    if a.cmd == "status":
        keep = ("updated_utc", "state", "stopping", "paused", "keys", "outcomes", "in_flight", "attempts_per_min", "terminal_keys_per_min",
                "latency_s_last_5m", "concurrency", "spend_usd", "provider_mix", "off_protocol_held", "parked_keys", "rerouted_keys", "eta")
        out = {"batch_gate_open": paths.gate_open(), "combined_progress": os.path.join(paths.live_root, "progress.json"), "runs": {}}
        for run in _runs(a):
            ld = paths.live(run)
            if not os.path.exists(os.path.join(ld, "state.sqlite")):
                out["runs"][run] = {"initialized": False}
                continue
            p = _read(os.path.join(ld, "progress.json"))
            r = readiness(ld)
            _spec, cfg, allowed = LV.live_settings(ld)
            con = sqlite3.connect(f"file:{os.path.join(ld, 'state.sqlite')}?mode=ro", uri=True, timeout=60)
            try:
                pk = LV.parked(con, cfg, allowed)
            finally:
                con.close()
            out["runs"][run] = {"staged": staged(ld), "last_run": LV.last_run(ld),
                                "auto_start": LV.auto_start_refusal(ld, allow_parked=True) or "allowed",
                                "parked_now": {"keys": len(pk), "reroutable": sum(1 for v in pk.values() if v["not_yet_tried"])},
                                "progress": {k: p.get(k) for k in keep}, "readiness": {k: v for k, v in r.items() if k != "fingerprint"}}
        print(json.dumps(out, indent=1, default=str))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
