"""PRODUCER-SIDE record writer — STDLIB ONLY (json, hashlib, uuid, os, socket, datetime, subprocess).

Importable on a compute node before ``uv sync``::

    python3 -S -c "import sisteer.registry.emit"

A producer appends JSONL record files to a spool on ITS OWN storage
(``<spool>/<campaign>/<run_dir>/<seq>-<record_type>.jsonl``), never to the database. The record
files are the system of record; a correction is a new record, never a rewrite. After every write
the writer calls ``push_records.sh`` (debounced to one push per 30 s per run dir; ``--final`` at the
end); a failed push is logged and the harvester's pull is the backstop, so no producer step ever
blocks on the grading host. ``SISTEER_REGISTRY_DISABLED=1`` makes every method a no-op (byte-identity
tests run the stores with and without a recorder).

Two variables pin the HOST PROBES (:func:`spool_root`, :func:`substrate_from_env`) from outside the process, for a
producer that runs as a subprocess (``python3 -S scripts/registry/smoke_producer.py``, the job shell's ``emit_cli``
lines), which no in-process monkeypatch reaches: ``SISTEER_REGISTRY_SUBSTRATE=<name>`` declares the substrate outright
(validated against the vocabulary; an unknown name is one stderr notice and the probes decide) and
``SISTEER_REGISTRY_PROBE_ROOT=<dir>`` looks every fixed host path up under ``<dir>`` instead of ``/``. Neither is set on
a real host; neither ever raises into a workload. A cluster job learns its substrate from the site instead:
``scripts/registry/registry_env.sh`` exports ``SISTEER_REGISTRY_SUBSTRATE`` and the site's Slurm-cluster map
``SISTEER_SLURM_SUBSTRATES`` (:data:`SITE_MAP_ENV`), and no cluster name or shared root is spelled in this module.

Ids: ``record_id`` is uuid4 (producers); ``run_id`` comes from :mod:`identity` so the adapter that
later reads the same store mints the same id.
"""
from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import sys
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import identity
from .identity import run_dir_name
from .records import ITEM_SCALE_RECORD_TYPES, make_record, record_file_name, run_patch_extra_json, utc_now, write_records
from .vocab import SUBSTRATES, bon_metric_refusal, counted_metric_refusal

DISABLED_ENV = "SISTEER_REGISTRY_DISABLED"
SPOOL_ENV = "SISTEER_REGISTRY_SPOOL"
PUSH_SCRIPT_ENV = "SISTEER_REGISTRY_PUSH"
DEBOUNCE_S = 30.0
#: The run-overrides file: a `.env` of `OVERRIDE_*` keys naming this run's deliberate
#: deviations from the paper's defaults. Every producer :class:`RecordWriter` reads it ONCE at construction and parks
#: ``{"id", "sha256", "keys": {...}}`` under ``payload.extra.run_overrides`` of EVERY record it writes (``run``, ``run_state``,
#: ``object``, ``fit_search``, ``fit_trial``, ``judgment``, ``item_result`` ...) -- no schema change, the fold parks it in
#: ``extra_json``, where ``json_extract(extra_json, '$.run_overrides.id')`` reads it. Unset (the paper's defaults) → no stamp.
RUN_OVERRIDES_ENV = "SISTEER_RUN_OVERRIDES"
RUN_OVERRIDES_KEY = "run_overrides"
#: The site label a `slurm` (generic third-cluster) substrate MUST carry (migration 0018): parked as
#: ``payload.extra.cluster_label`` on every record the writer emits; the writer refuses `slurm` without it, by name.
CLUSTER_LABEL_ENV = "SISTEER_REGISTRY_CLUSTER_LABEL"
CLUSTER_LABEL_KEY = "cluster_label"
GENERIC_SLURM_SUBSTRATE = "slurm"

#: Two overrides a SUBPROCESS can carry (module docstring): a gate container once found the bare-file smoke producer
#: reporting the container's own provisioned spool while its test asserted ``grading_host`` (2026-09-17) -- the
#: ``unprovisioned_host`` fixture redirects the module constants in-process, which a ``python3 -S`` child never sees, so
#: the same redirection is spelled as environment variables here. Neither ever raises into a workload.
SUBSTRATE_ENV = "SISTEER_REGISTRY_SUBSTRATE"      # the substrate by name (one of vocab.SUBSTRATES); wins over every probe
PROBE_ROOT_ENV = "SISTEER_REGISTRY_PROBE_ROOT"    # a directory the fixed host paths below are looked up UNDER, instead of /

#: The fixed host paths :func:`spool_root` and :func:`substrate_from_env` probe, in probe order — module constants
#: rather than literals inside the functions, so a test can point EVERY probe under its own ``tmp_path`` (the
#: ``unprovisioned_host`` fixture) and assert against nothing on the real host. ``{user}`` is filled from ``$USER`` at
#: probe time. Every probe goes through :func:`probe_path`, so ``$SISTEER_REGISTRY_PROBE_ROOT`` relocates them all for a
#: subprocess. No shared root is fixed here: :func:`shared_bases` adds the roots the site map names after these.
SHARED_BASES: Tuple[Tuple[str, str], ...] = ()       # (substrate, per-user shared root), probed before the site map's
#: The site's Slurm-cluster map: entries ``<slurm cluster name>=<label>`` or ``<slurm cluster name>=<label>@<shared root>``,
#: separated by spaces or commas. The cluster a job runs on (``$SLURM_CLUSTER_NAME``, matched whole and case-insensitively)
#: names its registry label, and an entry's shared root (absolute; ``{user}`` filled from ``$USER``) joins
#: :func:`shared_bases`. ``scripts/registry/registry_env.sh`` exports it on a cluster from the site config; unset, nothing
#: is mapped.
SITE_MAP_ENV = "SISTEER_SLURM_SUBSTRATES"
#: Exported by a cluster helper installed before the site map existed (``registry_env.sh`` then named the host this way and
#: declared no substrate). A Slurm job that carries it and none of the variables above sourced that older helper: it gets the
#: generic ``slurm`` and is refused by name until the helpers are installed again (``reproduce/00_setup.sh --what helpers``).
PREVIOUS_HELPER_ENV = "SISTEER_REGISTRY_HOST"
HOME_REGISTRY = "~/registry"                          # the grading host's registry, expanded through $HOME

_NOTICED: set = set()


def _notice_once(msg: str) -> None:
    """One ``[registry]`` stderr line per distinct message per process: a misconfigured override is said once, never raised."""
    if msg not in _NOTICED:
        _NOTICED.add(msg)
        print(f"[registry] {msg}", file=sys.stderr, flush=True)


class MetricRefused(ValueError):
    """A `run_metric` row refused at EMISSION by the best-of-N guard (:func:`vocab.bon_metric_refusal`) or by the denominator guard
    (:func:`vocab.counted_metric_refusal`: a counted row with ``n = 0`` is a non-measurement; a boolean or unreadable ``n`` is malformed
    and refused by name, never skipped): nothing of the batch is written. Emit-side only -- the validator is untouched, so the live rows
    fold as they always did."""


class SubstrateRefused(ValueError):
    """A producer named a substrate the vocabulary does not admit, or the generic `slurm` without its `cluster_label`:
    nothing is written and the refusal names the value -- never an alias to `grading_host`. A hook that builds a writer catches it as it
    catches every failure (one ``[registry]`` line; the workload continues)."""


def check_substrate(substrate: Optional[str], cluster_label: Optional[str] = None) -> Optional[str]:
    """The `cluster_label` a writer must stamp for ``substrate``: None for a named substrate; the label (``cluster_label`` or
    ``$SISTEER_REGISTRY_CLUSTER_LABEL``) for the generic `slurm`. Raises :class:`SubstrateRefused` for a value outside
    :data:`SUBSTRATES` and for `slurm` without a label."""
    name = str(substrate or "")
    if name not in SUBSTRATES:
        raise SubstrateRefused(f"substrate {name!r} is not one of {'|'.join(SUBSTRATES)}: a third Slurm cluster declares "
                               f"SISTEER_REGISTRY_SUBSTRATE={GENERIC_SLURM_SUBSTRATE} (--substrate {GENERIC_SLURM_SUBSTRATE}) with "
                               f"{CLUSTER_LABEL_ENV}=<its label> (migration 0018); an unknown substrate is never aliased to grading_host")
    if name != GENERIC_SLURM_SUBSTRATE:
        return None
    label = (cluster_label or os.environ.get(CLUSTER_LABEL_ENV) or "").strip()
    if not label:
        raise SubstrateRefused(f"substrate {GENERIC_SLURM_SUBSTRATE!r} needs a site label: set {CLUSTER_LABEL_ENV}=<label> (recorded as "
                               f"extra_json.cluster_label on every record), or name the cluster's registry label in {SITE_MAP_ENV}; "
                               "scripts/registry/registry_env.sh exports both from the site config, and a cluster whose helpers predate "
                               "it installs them again (reproduce/00_setup.sh --what helpers)")
    return label


_UNQUOTED_COMMENT = re.compile(r"\s+#")
_EMPTY_THEN_COMMENT = re.compile(r"^\s+#")


def env_value(raw: str) -> str:
    """The value of a ``KEY=value`` line's right-hand side as the shell reads it, for the file format the registry accepts (``KEY=value``
    with optional single or double quotes, an optional inline comment). ``raw`` is the text after ``=``:

    * **empty, then a comment** (``KEY=   # use the default``; a tab counts): the empty string. The comment boundary is read BEFORE the
      surrounding whitespace is dropped -- stripped first, ``# use the default`` would stand alone and read as a value beginning with
      ``#`` -- and it needs the whitespace: ``KEY=#literal`` is the literal ``#literal``, as the shell reads it;
    * **quoted** (``'...'`` or ``"..."``): the quoted span is the value, a ``#`` inside it literal (``'64 # kept'`` -> ``64 # kept``;
      ``''   # empty`` -> the empty string); what follows the closing quote is read by the unquoted rule and appended (``'a'b`` -> ``ab``),
      so a trailing comment after the quote is dropped;
    * **unquoted**: the value ends at the first ``#`` PRECEDED BY WHITESPACE (``64   # unchanged`` -> ``64``; a tab counts); a ``#`` with no
      whitespace before it is part of the value (``64#notacomment``, ``v#x # trailing`` -> ``v#x``), as the shell reads it.

    Outside the format, and so outside this rule: backslash escapes, partial quoting (``a' # 'b``), ``$VAR`` expansion and an unquoted
    value containing whitespace (``KEY=a b``, which the shell reads as a command). Stripping EVERY ``#...`` tail was rejected: the shell
    keeps ``v#x``, and a quoted literal ``#``. The old reading kept the whole right-hand side, so every value on a commented line folded
    with its comment text (``"64                  # unchanged: the paper's"`` on every stamped live run row); ``registry reconcile
    env-comments`` restates a stored stamp from its overrides file read by this rule, and ``tests/test_emit_overrides_stamp.py`` pins this
    reader equal to a ``source``d shell's."""
    if _EMPTY_THEN_COMMENT.match(raw):
        return ""
    v = raw.strip()
    if v and v[0] in "'\"":
        close = v.find(v[0], 1)
        if close > 0:
            return v[1:close] + _unquoted_env_value(v[close + 1:])
    return _unquoted_env_value(v)


def _unquoted_env_value(v: str) -> str:
    """The unquoted rule of :func:`env_value`: cut at the first ``#`` preceded by whitespace, then strip."""
    return _UNQUOTED_COMMENT.split(v, 1)[0].strip()


def _parse_env_file(path: str, value_of: Callable[[str], str] = env_value) -> Dict[str, str]:
    """``KEY=value`` lines of a shell `.env`: comment and blank lines skipped, a leading ``export`` stripped, each value read by
    ``value_of`` -- :func:`env_value` (quotes stripped, an inline comment after an unquoted or a quoted value dropped, a ``#`` with no
    whitespace before it or inside quotes kept). The line handling is the one thing every reading of a file shares: ``registry reconcile
    env-comments`` reads a file by the reader BEFORE the fix too (``reconcile.historical_env_value``), as the evidence that a stored value
    is that reader's artifact and not a later statement, and must walk the lines exactly as the writer did."""
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            k, _, v = line.partition("=")
            k = k.strip()
            if k:
                out[k] = value_of(v)
    return out


def run_overrides_stamp(path: Optional[str] = None) -> Optional[Dict[str, object]]:
    """``{"id", "path", "sha256", "keys"}`` for the overrides file ``$SISTEER_RUN_OVERRIDES`` (or ``path``); None when unset (the
    paper's defaults). ``id`` is the file's ``OVERRIDE_ID`` (else its stem); ``keys`` every ``OVERRIDE_*`` value verbatim. An
    unreadable file is a refusal, never a silent no-stamp: a run that names an overrides file it cannot read is misconfigured."""
    p = path if path is not None else os.environ.get(RUN_OVERRIDES_ENV, "")
    p = (p or "").strip()
    if not p:
        return None
    p = os.path.expanduser(p)
    with open(p, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    kv = _parse_env_file(p)
    keys = {k: v for k, v in kv.items() if k.startswith("OVERRIDE_")}
    ident = keys.get("OVERRIDE_ID") or os.path.splitext(os.path.basename(p))[0]
    return {"id": ident, "path": os.path.basename(p), "sha256": digest, "keys": keys}


def probe_path(path: str) -> str:
    """Where the probes look for a fixed host path: ``path`` itself (``~`` expanded), or — with
    ``$SISTEER_REGISTRY_PROBE_ROOT`` set — the same path under that root (``/vol/registry`` → ``<root>/vol/registry``,
    ``~/registry`` → ``<root>/<$HOME>/registry``), so a subprocess can be pointed at a tmp host, or at an empty one."""
    path = os.path.expanduser(path)
    root = os.environ.get(PROBE_ROOT_ENV, "").strip()
    if not root:
        return path
    return os.path.join(root, os.path.abspath(path).lstrip("/"))


def substrate_override() -> Optional[str]:
    """``$SISTEER_REGISTRY_SUBSTRATE`` when it names a substrate of the vocabulary (case-insensitive), else None: an
    unknown name is one stderr notice and the host probes decide, exactly as if it were unset — a typo never stops a job."""
    raw = os.environ.get(SUBSTRATE_ENV)
    if raw is None or not raw.strip():
        return None
    name = raw.strip().lower()
    if name in SUBSTRATES:
        return name
    _notice_once(f"{SUBSTRATE_ENV}={raw.strip()!r} is not a substrate ({'|'.join(SUBSTRATES)}); ignored, the host probes decide")
    return None


def spool_root(default: Optional[str] = None) -> Optional[str]:
    """The producer's spool, in this order: ``$SISTEER_REGISTRY_SPOOL``; ``<base>/registry/records`` for
    the first base of :func:`shared_bases` that exists AND whose ``registry/`` directory has been provisioned
    (``scripts/registry/install_cluster.sh``); ``<HOME_REGISTRY>/records`` on the grading host when ``~/registry`` exists;
    else ``default`` (None → emission disabled with a one-line notice). An unprovisioned host
    emits nothing. Every path is looked up through :func:`probe_path`, and the spool returned is the looked-up path —
    under ``$SISTEER_REGISTRY_PROBE_ROOT`` when that is set."""
    env = os.environ.get(SPOOL_ENV)
    if env:
        return env
    user = os.environ.get("USER", "")
    for _substrate, template in shared_bases():
        base = probe_path(template.replace("{user}", user))
        if user and os.path.isdir(os.path.join(base, "registry")):
            return os.path.join(base, "registry", "records")
    home_reg = probe_path(HOME_REGISTRY)
    if os.path.isdir(home_reg):
        return os.path.join(home_reg, "records")
    return default


def site_map(raw: Optional[str] = None) -> List[Tuple[str, str, Optional[str]]]:
    """``[(slurm cluster name, label, shared root or None), ...]`` of :data:`SITE_MAP_ENV` (or ``raw``), in the order written;
    the name lower-cased. An entry that does not parse (no ``=``, an empty side, a shared root that is not absolute) is one
    stderr notice and skipped, exactly as an unknown ``$SISTEER_REGISTRY_SUBSTRATE`` is: a typo never stops a job. The label
    is taken as written: one outside the vocabulary reaches :func:`check_substrate`, which refuses it by name."""
    text = os.environ.get(SITE_MAP_ENV, "") if raw is None else raw
    out: List[Tuple[str, str, Optional[str]]] = []
    for entry in re.split(r"[\s,]+", text.strip()):
        if not entry:
            continue
        name, eq, rest = entry.partition("=")
        label, at, root = rest.partition("@")
        name, label, root = name.strip().lower(), label.strip(), root.strip()
        if not eq or not name or not label or (at and not root.startswith("/")):
            _notice_once(f"{SITE_MAP_ENV}: entry {entry!r} is not <slurm cluster name>=<label>[@<absolute shared root>]; skipped")
            continue
        out.append((name, label, root or None))
    return out


def shared_bases() -> Tuple[Tuple[str, str], ...]:
    """The ``(substrate, per-user shared root)`` pairs :func:`spool_root` and :func:`substrate_from_env` probe, in order:
    :data:`SHARED_BASES`, then each site-map entry that names a shared root (:func:`site_map`)."""
    return tuple(SHARED_BASES) + tuple((label, root) for _name, label, root in site_map() if root)


def substrate_from_env() -> str:
    """The producing substrate, in this order: ``$SISTEER_REGISTRY_SUBSTRATE`` when it names one (:func:`substrate_override`);
    inside a Slurm job (``$SLURM_CLUSTER_NAME`` set) at a site that declares a map (:data:`SITE_MAP_ENV`), the entry named by
    the job's cluster, else the generic ``slurm`` before any shared root is probed, so another entry's shared root mounted on the
    job's machine never names it; the first base of :func:`shared_bases` whose per-user root exists; inside a Slurm job at a site
    that declares a cluster label (``$SISTEER_REGISTRY_CLUSTER_LABEL``) or that sourced the older helper
    (:data:`PREVIOUS_HELPER_ENV`), the generic ``slurm``; else RunPod; else the grading host. :func:`check_substrate` refuses the
    generic ``slurm`` by name without its label, so a cluster the map does not name is refused rather than mislabelled. The paths
    go through :func:`probe_path`. A Slurm job with no site information at all (the test suite's own jobs, a host the site never
    configured) falls through to the last two."""
    declared = substrate_override()
    if declared:
        return declared
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "").strip().lower()
    entries = site_map()
    if cluster and entries:
        for name, label, _root in entries:
            if name == cluster:
                return label
        return GENERIC_SLURM_SUBSTRATE
    user = os.environ.get("USER", "")
    for substrate, template in shared_bases():
        if user and os.path.isdir(probe_path(template.replace("{user}", user))):
            return substrate
    if cluster and (os.environ.get(CLUSTER_LABEL_ENV, "").strip() or os.environ.get(PREVIOUS_HELPER_ENV, "").strip()):
        return GENERIC_SLURM_SUBSTRATE
    if os.environ.get("RUNPOD_POD_ID"):
        return "runpod"
    return "grading_host"


def git_sha(cwd: Optional[str] = None) -> str:
    """The producing checkout's HEAD sha, else the literal ``unverified``."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode == 0 and len(sha) == 40:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unverified"


#: Where a producer finds the name of the session it runs in, the first one set wins: the registry's own variable, then the
#: one the session manager hands every session it starts, read the way ``SLURM_JOB_ID`` and ``RUNPOD_POD_ID`` are. A session
#: that only the manager named still names its records. The harvester keeps an equal tuple; a test pins that every reader of a
#: session name agrees.
SESSION_ENV_VARS: Tuple[str, ...] = ("SISTEER_SESSION_NAME", "ROMP_SESSION_NAME")


def session_from_env() -> Optional[str]:
    """The producing session's name: the first of :data:`SESSION_ENV_VARS` set to a non-empty value, else ``None``."""
    for var in SESSION_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


def producer_block(script: str, code_sha: Optional[str] = None, session: Optional[str] = None,
                   substrate: Optional[str] = None) -> Dict[str, object]:
    return {
        "script": script,
        "code_sha": code_sha or git_sha(),
        "session": session if session is not None else session_from_env(),
        "host": socket.gethostname(),
        "substrate": substrate or substrate_from_env(),
    }


# --------------------------------------------------------------------------------------------
# Hardware at emission time: the job's OWN environment first, sacct as the fallback
# --------------------------------------------------------------------------------------------
#: GPU-count variables in probe order. ``SLURM_GPUS_ON_NODE`` is a COUNT (``8``, or typed ``h200:8``); the others are
#: comma lists of device ids (``SLURM_JOB_GPUS`` / ``SLURM_STEP_GPUS`` ``0,1,2``; ``CUDA_VISIBLE_DEVICES`` indices or
#: GPU-UUIDs). Slurm sets the first three in every batch step of a GPU allocation; the last is the harness's own view.
GPU_COUNT_ENV_LADDER: Tuple[str, ...] = ("SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")
NVIDIA_SMI_ARGV: Tuple[str, ...] = ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
#: `-X` allocations only (no steps), `-n` no header, `-P` parsable: one ``AllocTRES`` line per job id
SACCT_ARGV: Tuple[str, ...] = ("sacct", "-X", "-n", "-P", "-o", "AllocTRES", "-j")
SACCT_SOURCE = "sacct_alloc_tres"
NVIDIA_SMI_SOURCE = "nvidia_smi"


def _count_of(var: str, raw: str) -> Optional[int]:
    v = raw.strip()
    if not v:
        return None
    if var == "SLURM_GPUS_ON_NODE":
        # "8" | "h200:8" | "h200:4,a100:2" (typed, several types) — the sum of the counts
        total = 0
        for part in v.split(","):
            tail = part.rsplit(":", 1)[-1].strip()
            if not tail.isdigit():
                return None
            total += int(tail)
        return total
    ids = [x for x in (x.strip() for x in v.split(",")) if x]
    return len(ids) or None


def gpu_count_from_env(env: Optional[Mapping[str, str]] = None) -> Tuple[Optional[int], Optional[str]]:
    """``(count, variable)`` from the first variable of :data:`GPU_COUNT_ENV_LADDER` that is set AND parses; a set
    variable that does not parse is skipped, never guessed. ``(None, None)`` when nothing is set."""
    env = os.environ if env is None else env
    for var in GPU_COUNT_ENV_LADDER:
        raw = env.get(var)
        if raw is None:
            continue
        n = _count_of(var, raw)
        if n is not None:
            return n, var
    return None, None


def _run_quiet(argv: Sequence[str], timeout_s: float) -> Optional[str]:
    """stdout of a tool that may be absent, slow or broken; None on ANY failure (this runs inside a job)."""
    try:
        out = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def gpu_type_from_nvidia_smi(timeout_s: float = 5.0) -> Optional[str]:
    """The first GPU's name from ``nvidia-smi --query-gpu=name --format=csv,noheader``; None without a working nvidia-smi."""
    out = _run_quiet(NVIDIA_SMI_ARGV, timeout_s)
    if not out:
        return None
    for line in out.splitlines():
        if line.strip():
            return line.strip()
    return None


def parse_alloc_tres(text: str) -> Dict[str, object]:
    """``AllocTRES`` (``billing=8,cpu=64,gres/gpu=4,gres/gpu:h200=4,mem=512G,node=1``) → ``{"gpu_count", "gpu_type"}``:
    the untyped ``gres/gpu=N`` entry wins for the count, else the sum of the typed ``gres/gpu:<type>=N`` entries; the type
    is the one typed name when exactly one type is allocated. Empty dict when no gpu entry is present."""
    untyped: Optional[int] = None
    typed: Dict[str, int] = {}
    for part in (text or "").replace("\n", ",").split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if not val.isdigit():
            continue
        if key == "gres/gpu":
            untyped = int(val)
        elif key.startswith("gres/gpu:"):
            typed[key[len("gres/gpu:"):]] = int(val)
    out: Dict[str, object] = {}
    count = untyped if untyped is not None else (sum(typed.values()) if typed else None)
    if count is not None:
        out["gpu_count"] = count
    if len(typed) == 1:
        out["gpu_type"] = next(iter(typed))
    return out


def hardware_from_sacct(job_id: Optional[str], timeout_s: float = 10.0) -> Dict[str, object]:
    """The cluster-side fallback reader: ``sacct -X -n -P -o AllocTRES -j <job>`` parsed by :func:`parse_alloc_tres`,
    with ``"source": "sacct_alloc_tres"``; ``{}`` without a job id, without sacct, or on any failure."""
    if not job_id or not str(job_id).strip():
        return {}
    out = _run_quiet(list(SACCT_ARGV) + [str(job_id).strip()], timeout_s)
    if not out or not out.strip():
        return {}
    parsed = parse_alloc_tres(out.strip().splitlines()[0])
    if parsed:
        parsed["source"] = SACCT_SOURCE
    return parsed


def hardware_from_env(env: Optional[Mapping[str, str]] = None, job_id: Optional[str] = None, *, sacct: bool = True,
                      nvidia_smi: bool = True, timeout_s: float = 5.0) -> Dict[str, object]:
    """What the job can say about its GPUs at emission time, never raising: ``gpu_count`` from the env ladder
    (``gpu_count_source`` = the variable), else from sacct's ``AllocTRES`` for ``job_id`` / ``$SLURM_JOB_ID``
    (``sacct_alloc_tres``); ``gpu_type`` from nvidia-smi (``nvidia_smi``), else sacct's one typed gres name. Absent
    facts are None with a None source — a caller files nothing for them rather than a guess."""
    env = os.environ if env is None else env
    out: Dict[str, object] = {"gpu_count": None, "gpu_count_source": None, "gpu_type": None, "gpu_type_source": None}
    try:
        n, var = gpu_count_from_env(env)
        if n is not None:
            out["gpu_count"], out["gpu_count_source"] = n, var
        if nvidia_smi:
            t = gpu_type_from_nvidia_smi(timeout_s)
            if t:
                out["gpu_type"], out["gpu_type_source"] = t, NVIDIA_SMI_SOURCE
        if sacct and (out["gpu_count"] is None or out["gpu_type"] is None):
            acct = hardware_from_sacct(job_id or env.get("SLURM_JOB_ID"), timeout_s=max(timeout_s, 10.0))
            if out["gpu_count"] is None and acct.get("gpu_count") is not None:
                out["gpu_count"], out["gpu_count_source"] = int(acct["gpu_count"]), SACCT_SOURCE
            if out["gpu_type"] is None and acct.get("gpu_type"):
                out["gpu_type"], out["gpu_type_source"] = str(acct["gpu_type"]), SACCT_SOURCE
    except Exception:  # noqa: BLE001 — a hardware probe never reaches the workload
        pass
    return out


def hardware_note(hw: Mapping[str, object]) -> Optional[str]:
    """One line naming where each filled hardware fact came from, for the `running` state's note."""
    parts = []
    if hw.get("gpu_count") is not None:
        parts.append(f"gpu_count {hw['gpu_count']} from {hw.get('gpu_count_source')}")
    if hw.get("gpu_type"):
        parts.append(f"gpu_type from {hw.get('gpu_type_source')}")
    return "; ".join(parts) or None


class RecordWriter:
    """Append records for ONE run (or one dimension/object batch) to its own record directory."""

    def __init__(self, spool: Optional[str], campaign: str, run_id: str, producer: Mapping[str, object],
                 push_script: Optional[str] = None, disabled: Optional[bool] = None,
                 payload_extra: Optional[Mapping[str, object]] = None):
        self.disabled = bool(int(os.environ.get(DISABLED_ENV, "0"))) if disabled is None else disabled
        if spool is None and not self.disabled:
            self.disabled = True
            self.disabled_reason = "no spool configured (set SISTEER_REGISTRY_SPOOL)"
        else:
            self.disabled_reason = ""
        self.spool = spool
        self.campaign = campaign
        self.run_id = run_id
        self.producer = dict(producer)
        self.push_script = push_script or os.environ.get(PUSH_SCRIPT_ENV)
        self.dir = os.path.join(spool, campaign, run_dir_name(run_id)) if spool else None
        self._seq = None
        self._last_utc = ""
        self.written: List[str] = []
        # THE CENTRAL STAMPS (every producer writer is constructed here -- emit_cli.register_from_config,
        # write_planned_run, hooks.JudgingScope, hooks.StoreRecorder, the Modal launchers -- so a stamp bolted onto a helper would reach
        # none of them): the run-overrides file and the third-cluster label are read ONCE per writer and merged into every payload's
        # ``extra`` by write_many. An explicit ``payload_extra=`` wins key by key. A disabled writer (the harvester's own records, the
        # adapters, the suite) stamps nothing and checks nothing: those processes never set the variables, and a refusal must not reach them.
        self.payload_extra: Dict[str, object] = dict(payload_extra or {})
        if not self.disabled:
            if self.producer.get("substrate") is not None:      # a producer that names a substrate names one the vocabulary admits
                label = check_substrate(self.producer.get("substrate"), self.payload_extra.get(CLUSTER_LABEL_KEY))   # SubstrateRefused by name
                if label is not None:
                    self.payload_extra.setdefault(CLUSTER_LABEL_KEY, label)
            if RUN_OVERRIDES_KEY not in self.payload_extra:
                stamp = run_overrides_stamp()
                if stamp is not None:
                    self.payload_extra[RUN_OVERRIDES_KEY] = stamp

    @property
    def last_written_utc(self) -> str:
        """The last ``written_utc`` this writer stamped ('' before the first): what a caller carries to the next writer of the same key
        (:meth:`not_before`) when it builds one writer per call, so the key's records stay strictly ordered across writers."""
        return self._last_utc

    def not_before(self, written_utc: str) -> None:
        """Keep every stamp this writer issues STRICTLY ABOVE ``written_utc`` (a lexicographic UTC-ms stamp; '' is a no-op). The fold
        orders a key's records by their own stamps and :meth:`_stamp` is monotone only WITHIN a writer, so two writers built within one
        millisecond for the same key tie -- and tied records fold in sha order, arbitrarily (sisteer.registry.emit_hpo's open / progress /
        close records of one search did, 2026-10-05). A caller that builds a fresh writer per record hands it the previous writer's
        :attr:`last_written_utc`."""
        if written_utc and written_utc > self._last_utc:
            self._last_utc = written_utc

    def _stamp(self) -> str:
        """A written_utc that is STRICTLY monotone within this writer and NEVER ahead of the wall clock:
        two records this writer emits in one millisecond would otherwise tie on ``written_utc`` and be
        ordered by sha — arbitrarily — so a ``judgment`` and the ``run_state judged`` that depends on it
        could fold in either order. The fold's order key is the record's own stamp, so the emitter keeps
        its emission order in it by WAITING for the next millisecond (never by stamping a future instant:
        the harvester's own facts carry its cycle clock, and a producer stamp ahead of it would fold
        after them). A writer emits tens of files per run, so the wait is at most tens of milliseconds."""
        now = utc_now()
        while now <= self._last_utc:
            time.sleep(0.0005)
            now = utc_now()
        self._last_utc = now
        return now

    # -- sequencing -------------------------------------------------------------------------
    #: Exclusive creates a single write attempts before giving up (each attempt re-lists the directory; a run directory gains a few
    #: files per pass, so a writer that loses this many races in a row is reporting something else -- a directory it cannot create in).
    SEQ_ATTEMPTS = 50

    def _dir_seq_high(self) -> int:
        """The highest sequence number any entry of the run directory carries RIGHT NOW: every entry's numeric head, a ``.gz`` twin the
        harvester renamed, another record type at the same number and a file another pass or the cluster mirror delivered included
        (0 for an absent or empty directory)."""
        high = 0
        if self.dir and os.path.isdir(self.dir):
            for fn in os.listdir(self.dir):
                head = fn.split("-", 1)[0]
                if head.isdigit():
                    high = max(high, int(head))
        return high

    def _next_seq(self) -> int:
        """The sequence number of the NEXT file this writer creates, allocated AT WRITE TIME:
        one above the greater of the directory's listing taken now and the last number this writer issued -- monotone within the writer,
        and never a number the directory acquired since the writer was built. Before this a writer listed the directory ONCE, lazily at
        its first write, and counted up from that cached census: a writer constructed early and used after a long phase re-issued a
        number the harvester's gz rename, an earlier pass or the cluster mirror had taken meanwhile (``000016-run_degeneration.jsonl``
        written while the log held ``000016-run_degeneration.jsonl.gz``, 2026-10-06), and the exclusive create in :meth:`write_many`
        cannot see a twin under another suffix or another record type at the same number. ``_seq`` stays the last number issued."""
        self._seq = max(self._dir_seq_high(), self._seq or 0) + 1
        return self._seq

    def write(self, record_type: str, payload: Mapping[str, object], gz: bool = False) -> Optional[str]:
        return self.write_many(record_type, [payload], gz=gz)

    def write_many(self, record_type: str, payloads: Sequence[Mapping[str, object]], gz: bool = False,
                   written_utc: Optional[str] = None) -> Optional[str]:
        """Write one record FILE holding one record per payload (stamped ``written_utc``, default now); returns
        the path (None when disabled)."""
        if self.disabled:
            return None
        # an explicit stamp (the claims writer stamps a batch with the claim's own clock) is the caller's
        # ordering to keep; the writer's own stamps stay strictly monotone after it
        stamp = written_utc if written_utc is not None else self._stamp()
        if written_utc is not None and self._last_utc < written_utc <= utc_now():
            self._last_utc = written_utc
        stamped = self._stamped_patch if record_type == "run_patch" else self._stamped
        recs = [make_record(record_type, stamped(p), self.producer, identity.producer_record_id(), stamp) for p in payloads]
        # a record file is immutable once written: the sequence number is allocated from a fresh listing of the directory at
        # each attempt (_next_seq) and the file is created exclusively (O_EXCL); when another writer of the same run directory
        # (the job shell's state records, the serving hook) created this very name between the listing and the create, the
        # next attempt lists again and takes the next number -- never append to a file the transport may already have copied
        path = None
        for _ in range(self.SEQ_ATTEMPTS):
            path = os.path.join(self.dir, record_file_name(self._next_seq(), record_type, gz=gz))
            try:
                written = write_records(path, recs, gz=gz, exclusive=True)
            except FileExistsError:
                continue
            self.written.append(written)
            return written
        raise RuntimeError(f"RecordWriter: could not create a record file under {self.dir} in {self.SEQ_ATTEMPTS} attempts (last tried {path})")

    def _stamped(self, payload: Mapping[str, object]) -> Dict[str, object]:
        """The payload with this writer's central stamps (:attr:`payload_extra`) merged under ``extra`` -- a key the payload's own
        ``extra`` already carries is the payload's (a producer that states its overrides explicitly is not overwritten)."""
        if not self.payload_extra:
            return dict(payload)
        out = dict(payload)
        extra = out.get("extra")
        merged: Dict[str, object] = dict(self.payload_extra)
        if isinstance(extra, Mapping):
            merged.update(extra)
        elif extra is not None:
            merged["extra"] = extra
        out["extra"] = merged
        return out

    def _stamped_patch(self, payload: Mapping[str, object]) -> Dict[str, object]:
        """A ``run_patch`` payload with this writer's central stamps merged under ``fields.extra_json`` -- the ONE place a patch's
        provenance folds: the validator reads a patch's ``fields`` and ``verify`` only, so a top-level
        ``extra`` on a ``run_patch`` is never folded, and the stamps :meth:`_stamped` parks there on every other kind were lost on
        every patch (1,634 records on 668 runs by the audit). ``fields.extra_json`` is popped into the run's ``extra_json`` by the
        fold's additive merge, exactly as the native hook's own ``extra_json`` keys are. A key the payload's own ``fields.extra_json``
        already carries is the payload's; ``fields`` is created when the patch carried only ``verify``. A payload's own top-level
        ``extra`` is left where it is (the caller's, and documented as not folded), never moved.

        A ``fields`` that is not an object, or a ``fields.extra_json`` that is not a JSON object -- a serialized array or scalar,
        text that is not JSON, a non-mapping (:func:`records.run_patch_extra_json`) -- is refused HERE by name: ``ValueError``,
        nothing written, whether or not this writer has stamps. Such a value carries no keys to merge the stamps into, and the
        validator refuses the same shape before the fold, so writing it would only file a refusal. Earlier this method handed such
        a payload on UNSTAMPED, trusting the validator to refuse it -- and the validator's merge dropped the value silently
        instead, so a malformed correction folded its typed columns without the writer's ``run_overrides`` / ``cluster_label``
        stamps."""
        out = dict(payload)
        fields = out.get("fields")
        if fields is not None and not isinstance(fields, Mapping):
            raise ValueError(f"run_patch fields must be an object, got {type(fields).__name__}; nothing written")
        fields = dict(fields or {})
        own, reason = run_patch_extra_json(fields.get("extra_json"))
        if reason:
            raise ValueError(f"{reason}; nothing written")
        if not self.payload_extra:
            return out
        merged: Dict[str, object] = dict(self.payload_extra)
        if own:
            merged.update(own)
        fields["extra_json"] = merged
        out["fields"] = fields
        return out

    def _with_run(self, payload: Mapping[str, object]) -> Dict[str, object]:
        """A row payload with this writer's run_id filled in unless it names one."""
        return dict(payload, run_id=payload.get("run_id", self.run_id))

    def emit_rows(self, record_type: str, rows: Sequence[Mapping[str, object]], gz: bool = False) -> Optional[str]:
        """One record file of run-keyed rows (item_result, item_flags, run_metric, run_degeneration, ...). An
        item-scale row this writer emits is NATIVE (a producer's spool, never ``records/adapters/``), so its
        revision tag — part of the item natural key — is the '' sentinel unless the caller set one."""
        full = [self._with_run(r) for r in rows]
        if record_type in ITEM_SCALE_RECORD_TYPES:
            for r in full:
                r.setdefault("source_revision_id", "")
        return self.write_many(record_type, full, gz=gz)

    # -- typed emitters -----------------------------------------------------------------------
    def emit_run(self, run_payload: Mapping[str, object]) -> Optional[str]:
        payload = dict(run_payload)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("campaign", self.campaign)
        return self.write("run", payload)

    def emit_state(self, state: str, evidence_kind: str, evidence_uri: str = "", event_utc: Optional[str] = None,
                   scheduler_state: Optional[str] = None, note: Optional[str] = None) -> Optional[str]:
        payload = {"run_id": self.run_id, "state": state, "event_utc": event_utc or utc_now(),
                   "evidence_kind": evidence_kind, "evidence_uri": evidence_uri}
        if scheduler_state is not None:
            payload["scheduler_state"] = scheduler_state
        if note is not None:
            payload["note"] = note
        return self.write("run_state", payload)

    def emit_patch(self, fields: Optional[Mapping[str, object]] = None,
                   verify: Optional[Mapping[str, object]] = None) -> Optional[str]:
        payload: Dict[str, object] = {"run_id": self.run_id}
        if fields:
            payload["fields"] = dict(fields)
        if verify:
            payload["verify"] = dict(verify)
        return self.write("run_patch", payload)

    def emit_items(self, rows: Sequence[Mapping[str, object]], gz: bool = True) -> Optional[str]:
        return self.emit_rows("item_result", rows, gz=gz)

    def emit_item_flags(self, rows: Sequence[Mapping[str, object]], gz: bool = True) -> Optional[str]:
        return self.emit_rows("item_flags", rows, gz=gz)

    def emit_judgment(self, judgment_payload: Mapping[str, object]) -> Optional[str]:
        return self.write("judgment", self._with_run(judgment_payload))

    def emit_metrics(self, rows: Sequence[Mapping[str, object]]) -> Optional[str]:
        """``run_metric`` rows, after the best-of-N guard (:func:`vocab.bon_metric_refusal`) and the denominator guard
        (:func:`vocab.counted_metric_refusal`: a COUNTED row -- one carrying ``n`` -- with ``n = 0`` by numeric value is a non-measurement,
        never filed; a boolean or unreadable ``n`` is malformed and refused by name, since the validator folds a boolean into its integer
        and would store ``n = False`` as ``n = 0``; a row with no denominator is not counted and passes). Both are the `_stamped_patch`
        precedent: a rule applied at the writer, after the disabled check, never in the validator (a validator rule would alter the
        folded rows at every rebuild). One refused row refuses the whole batch by name, and nothing is written."""
        if self.disabled:
            return None
        for i, r in enumerate(rows):
            why = bon_metric_refusal(r) or counted_metric_refusal(r)
            if why:
                raise MetricRefused(f"run_metric row {i} of {len(rows)} refused at emission: {why}; nothing written")
        return self.emit_rows("run_metric", rows)

    def emit_degeneration(self, rows: Sequence[Mapping[str, object]]) -> Optional[str]:
        return self.emit_rows("run_degeneration", rows)

    def emit_object(self, object_payload: Mapping[str, object]) -> Optional[str]:
        return self.write("object", object_payload)

    def emit_artifact(self, artifact_payload: Mapping[str, object]) -> Optional[str]:
        return self.write("artifact", artifact_payload)

    # -- push -----------------------------------------------------------------------------------
    def push(self, final: bool = False) -> int:
        """Call ``push_records.sh <run_dir> [--final]``; ALWAYS returns (a failed push is the script's
        to log). Returns the script's exit code, 0 when there is nothing to do."""
        if self.disabled or not self.dir or not self.push_script or not os.path.isdir(self.dir):
            return 0
        stamp = os.path.join(self.dir, ".last_push")
        if not final and os.path.exists(stamp) and time.time() - os.path.getmtime(stamp) < DEBOUNCE_S:
            return 0
        args = [self.push_script, self.dir] + (["--final"] if final else [])
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
            return proc.returncode
        except (OSError, subprocess.SubprocessError):
            return 0


def writer_for_run(campaign: str, run_id: str, script: str, spool: Optional[str] = None,
                   code_sha: Optional[str] = None, session: Optional[str] = None) -> RecordWriter:
    """Convenience constructor: spool from the environment / shared root, producer block filled in."""
    return RecordWriter(spool or spool_root(), campaign, run_id, producer_block(script, code_sha, session))


class PlanRefused(ValueError):
    """A planned run whose identity is not fully resolved: nothing is written."""


def planned_run_records(plan: Mapping[str, object], producer: Mapping[str, object], now: Optional[str] = None):
    """The ``run`` (queued) + ``run_state queued`` payloads a plan yields, and its run id — pure."""
    for k in ("campaign", "producer_run_key_legacy", "producer_object_key_legacy", "eval_id", "instrument_id"):
        if not plan.get(k):
            named = f" for eval {plan.get('eval_id')!r}" if plan.get("eval_id") else ""
            raise PlanRefused(f"planned run{named} lacks {k!r}; it cannot be identified (identity.run_id needs campaign, "
                              "producer_run_key_legacy, producer_object_key_legacy, eval_id, instrument_id — an eval is never "
                              "named without its instrument)")
    ids = identity.run_id(str(plan["campaign"]), str(plan["producer_run_key_legacy"]), str(plan["producer_object_key_legacy"]),
                          str(plan["eval_id"]), str(plan["instrument_id"]))
    run_payload = {k: v for k, v in plan.items() if k not in ("submit_evidence_uri", "scheduler_state", "plan")}
    run_payload["run_id"] = ids
    run_payload.setdefault("lifecycle_state", "queued")
    now = now or utc_now()
    run_payload.setdefault("queued_utc", now)
    run_payload.setdefault("code_sha", producer.get("code_sha", "unverified"))
    run_payload.setdefault("substrate", producer.get("substrate", "grading_host"))
    run_payload.setdefault("substrate_source", "derived")
    state_payload = {"run_id": ids, "state": "queued", "event_utc": now, "evidence_kind": "submit_wrapper",
                     "evidence_uri": str(plan.get("submit_evidence_uri") or "")}
    if plan.get("scheduler_state") is not None:
        state_payload["scheduler_state"] = plan["scheduler_state"]
    return ids, run_payload, state_payload


def check_planned_run(plan: Mapping[str, object], producer: Optional[Mapping[str, object]] = None) -> str:
    """Strictness: validate the records a plan would produce WITHOUT
    writing; raises :class:`PlanRefused` naming the field (``GSM8K without saying which GSM8K``), returns
    the run id otherwise. Imports the validator lazily (it is not stdlib-only on its own imports)."""
    producer = producer or producer_block("registry plan-check")
    ids, run_payload, state_payload = planned_run_records(plan, producer)
    from . import validate as VAL
    for rt, payload in (("run", run_payload), ("run_state", state_payload)):
        rec = make_record(rt, payload, producer, identity.producer_record_id(), utc_now())
        v = VAL.validate_record(rec, "<planned-run>")
        if isinstance(v, VAL.Malformed):
            raise PlanRefused(f"planned {rt} refused (field {v.field}): {v.reason}")
    return ids


def write_planned_run(spool: str, plan: Mapping[str, object], script: str, code_sha: Optional[str] = None,
                      session: Optional[str] = None, strict: bool = True) -> List[str]:
    """Submit-time registration: a ``run`` (planned identity, ``lifecycle_state = queued``) + a
    ``run_state queued`` record. ``plan`` must carry the resolved identity fields; with ``strict``
    (the default) the records are validated FIRST and a plan whose instrument fields are not fully
    resolved raises :class:`PlanRefused` — nothing is written."""
    producer = producer_block(script, code_sha, session)
    if strict:
        check_planned_run(plan, producer)
    ids, run_payload, state_payload = planned_run_records(plan, producer)
    w = RecordWriter(spool, str(plan["campaign"]), ids, producer)
    out = []
    p = w.emit_run(run_payload)
    if p:
        out.append(p)
    p = w.write("run_state", state_payload)
    if p:
        out.append(p)
    return out
