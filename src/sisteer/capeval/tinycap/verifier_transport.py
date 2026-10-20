"""The BigCodeBench verifier's TRANSPORT into the Modal sandbox.

WHY. ``inspect_evals 0.18.0``'s ``bigcodebench.verify`` scorer runs the scored
completion plus the item's tests as ``sandbox().exec(["python", "-c", code])``
— the whole program rides in argv. The pinned ``inspect-sandboxes 0.5.0`` Modal
provider (``inspect_sandboxes.modal._modal.ModalSandboxEnvironment.exec``) hands
that argv straight to ``modal.Sandbox.exec``, whose ``_validate_exec_args``
refuses a CMD whose arguments total more than 65,536 (``ARG_MAX_BYTES``; it sums
``len(arg)`` over the arguments) BEFORE any RPC. A 16,384-token completion is
~73-79 KB of text on its own, so on the truncation ladder's rung 2 every
BigCodeBench item that runs to the cap is refused client-side, Inspect retries
the sample (the same deterministic overflow, another ~400 s of decode), and the
task is interrupted (ladder_report §4b: two llama8b_main items, 74,806 / 79,063
bytes, three draws each; BFCL never ran). Rung 1 (cap 4,096, ~18 KB) can never
hit it; ANY rung-2 item that runs to the cap can.

WHAT. :func:`install_exec_spill` wraps the provider class's ``exec`` so that a
``python -c <program>`` whose argv exceeds :data:`SPILL_THRESHOLD_BYTES` is
carried as a FILE instead: the program is written into the sandbox through the
EXEC channel with the program on STDIN (``sh -c 'cat > <path>'``, ``input=program``
— never argv, and never the provider's ``write_file``: that is inspect-sandboxes
0.5.0's ``Sandbox.mkdir`` + legacy ``open`` filesystem API, which Modal switched
OFF between the ladder run and the full wave of 2026-09-24 — "The legacy Sandbox
filesystem API is no longer supported" — and took a full BCB leg down on its one
>60 KB completion) at
``/tmp/sisteer_verify_<sha256(program)[:16]>.py`` and the ORIGINAL ``exec`` runs
``[python, <path>, *rest]`` with every keyword argument (input, cwd, env, user,
timeout, timeout_retry, concurrency) unchanged. Below the threshold the original
``exec`` is called with the original argv object, byte for byte — no rewrite, no
extra RPC. A command that is not ``python -c`` is never touched, whatever its
size. BigCodeBench's official evaluator never passes code as argv (it runs the
program text in a worker process); argv was inspect_evals' choice, and this
module moves only the transport, not the program, the image, the timeout or the
verdict grammar.

SEMANTIC DELTA of ``python <file>`` against ``python -c <program>``, recorded so
a reader can judge it: ``sys.argv[0]`` is the file path instead of ``'-c'``
(``unittest.main()`` reads ``sys.argv[1:]``, empty either way, so every test
class still runs); ``__main__.__file__`` is defined; ``sys.path[0]`` is the
file's directory (``/tmp``) instead of ``''`` (the cwd). The sandbox is fresh per
sample and ``/tmp`` holds only the spilled program, so the import-path change
resolves nothing new. The scorer's explanation — what ``inspect_served.bcb_row_facts``
reconstructs the verdict grammar from — is built from ``code``, not from the argv,
and is unchanged.

BYTE COUNT. :func:`argv_bytes` is the kernel's accounting — UTF-8 bytes of every
argument plus one NUL terminator each — which is never smaller than Modal's
``sum(len(arg))`` (code points), so every argv Modal would refuse at 65,536 is
spilled at the 60,000 threshold; the 5,536 margin is deliberate.

RECORD. Every spill appends one JSON line (``ts``, ``argv_bytes``,
``program_sha256``, ``path``) to the driver's per-store ledger
:data:`VERIFIER_SPILLS_FILENAME` — beside the usage ledger at the configuration's
out root for a SOLO run (:func:`ledger_path_for`); on a ROW ENGINE (the store's
manifest carries a ``row_engine`` block naming ``row_dir``) in the ROW dir instead,
``<row dir>/spills/<cell>__<store basename>__verifier_spills.jsonl``
(:func:`row_spill_ledger_path`, store-keyed like the generation markers): inode rule 2
fixes the per-cell store at exactly 40 and bounds the row dir, and a spill
into the store made a qwen s1 store count 41 — and logs
``[verifier-transport] spilled N bytes -> path`` at WARNING (the level Inspect
surfaces in the driver log by default). The launcher's REC carries
:func:`rec_field`, ``{"spill_threshold_bytes", "n_spilled"}``, read off that
ledger (0 when absent). Pure stdlib: no Modal or Inspect import here, so the
launcher process reads the ledger without the client venv.
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

__all__ = [
    "SPILL_THRESHOLD_BYTES", "MODAL_ARG_MAX_BYTES", "VERIFIER_SPILLS_FILENAME", "SPILL_PATH_TEMPLATE", "SPILL_WRITE_TIMEOUT_S",
    "PYTHON_ARGV0", "INSTALLED_MARKER", "LOG_PREFIX", "LEDGER_FIELDS",
    "ROW_SPILLS_SUBDIR",
    "argv_bytes", "is_python_c", "program_sha256", "spill_path", "ledger_path_for", "ledger_path_for_root",
    "row_spill_ledger_path", "row_engine_facts_of_manifest",
    "install_exec_spill", "read_spills", "n_spilled", "rec_field",
]

#: Spill at more than this many argv bytes (:func:`argv_bytes`). Below Modal's
#: 65,536 by a margin, above anything rung 1 produces (~18 KB at the 4,096 cap).
SPILL_THRESHOLD_BYTES = 60_000
#: ``modal.sandbox.ARG_MAX_BYTES`` — the limit ``_validate_exec_args`` enforces.
MODAL_ARG_MAX_BYTES = 65_536
#: The per-store ledger, at the configuration's out root beside ``usage_ledger.jsonl`` (a solo run).
VERIFIER_SPILLS_FILENAME = "verifier_spills.jsonl"
#: On a row engine the ledger lives in the ROW dir under this subdirectory, keyed by cell and store basename
#: (``<row dir>/spills/<cell>__<store basename>__verifier_spills.jsonl``) — never in the 40-inode store.
ROW_SPILLS_SUBDIR = "spills"
#: The program's path inside the sandbox; unique per program content.
SPILL_PATH_TEMPLATE = "/tmp/sisteer_verify_{sha16}.py"
#: the spill WRITE's own exec timeout (a `cat` of <= a few hundred KB; the verify exec keeps its caller's timeout)
SPILL_WRITE_TIMEOUT_S = 60
#: The interpreters a ``-c`` program is recognised under (the scorer says ``python``).
PYTHON_ARGV0 = ("python", "python3")
#: Set on the wrapping function; a class whose ``exec`` carries it is installed.
INSTALLED_MARKER = "__sisteer_exec_spill__"
LOG_PREFIX = "[verifier-transport]"
#: One ledger line's keys.
LEDGER_FIELDS = ("ts", "argv_bytes", "program_sha256", "path")

logger = logging.getLogger(__name__)


def argv_bytes(cmd: Sequence[str]) -> int:
    """The kernel's ARG_MAX accounting of ``cmd``: UTF-8 bytes of every argument
    plus one NUL terminator each (>= Modal's code-point sum)."""
    return sum(len(a.encode("utf-8")) for a in cmd) + len(cmd)


def is_python_c(cmd: Sequence[str]) -> bool:
    """``[python|python3, -c, <program>, ...]`` — the scorer's shape."""
    return len(cmd) >= 3 and cmd[0] in PYTHON_ARGV0 and cmd[1] == "-c"


def program_sha256(program: str) -> str:
    return hashlib.sha256(program.encode("utf-8")).hexdigest()


def spill_path(program: str) -> str:
    return SPILL_PATH_TEMPLATE.format(sha16=program_sha256(program)[:16])


def row_spill_ledger_path(row_dir: Union[str, Path], cell: str, store: str) -> Path:
    """``<row dir>/spills/<cell>__<store basename>__verifier_spills.jsonl`` — the row-engine home of a store's spill
    ledger (store-keyed: the in-run repeat runs the same cell into ``<suffix>-rep2``)."""
    store = str(store)
    if not store or "/" in store:
        raise ValueError(f"the store key must be the store root's basename (non-empty, no '/'), got {store!r}")
    return Path(row_dir) / ROW_SPILLS_SUBDIR / f"{cell}__{store}__{VERIFIER_SPILLS_FILENAME}"


def row_engine_facts_of_manifest(manifest_path: Union[str, Path]) -> Any:
    """``(row_dir, cell_id)`` when the store manifest at ``manifest_path`` carries a ``row_engine`` block naming a
    ``row_dir`` (tinycap-shared-standup: the driver copies the runner's record into the manifest before any task
    runs), else None — a solo store, no manifest yet, or an unreadable one (a solo-shaped decision, never a raise:
    the transport must install whatever the store looks like). Stdlib only."""
    p = Path(manifest_path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    block = doc.get("row_engine")
    row_dir = block.get("row_dir") if isinstance(block, dict) else None
    cell = doc.get("cell_id")
    if not row_dir or not isinstance(row_dir, str) or not cell or not isinstance(cell, str):
        return None
    return str(row_dir), str(cell)


def ledger_path_for(store: Any) -> Path:
    """The driver's spill ledger for ``store``. On a ROW ENGINE — the store's ``cells/<key>/manifest.json`` carries a
    ``row_engine`` block naming ``row_dir`` — it is :func:`row_spill_ledger_path` in the row dir (the store stays at
    exactly 40 inodes). Otherwise (a solo run) at the store's out root (``CellStore.sweep_dir`` — the
    directory that also holds ``usage_ledger.jsonl``, ``evidence.json`` and the driver logs; the launcher reads the
    REC field from there), as before. The runner's test doubles model a store as its ``cell_dir`` alone, so a store
    without ``sweep_dir`` ledgers beside its rows."""
    root = getattr(store, "sweep_dir", None)
    cell_dir = getattr(store, "cell_dir", None)
    if root is not None and cell_dir is not None:
        facts = row_engine_facts_of_manifest(Path(cell_dir) / "manifest.json")
        if facts is not None:
            return row_spill_ledger_path(facts[0], facts[1], Path(root).name)
    return Path(root if root is not None else cell_dir) / VERIFIER_SPILLS_FILENAME


def ledger_path_for_root(out_root: Union[str, Path]) -> Path:
    """The same decision from a store's OUT ROOT alone (the launcher's REC reader): the ONE ``cells/*/manifest.json``
    under it decides — a ``row_engine`` block with ``row_dir`` -> the row dir's ledger; none, several or no block ->
    the solo in-store path."""
    out_root = Path(out_root)
    manifests = sorted(out_root.glob("cells/*/manifest.json"))
    if len(manifests) == 1:
        facts = row_engine_facts_of_manifest(manifests[0])
        if facts is not None:
            return row_spill_ledger_path(facts[0], facts[1], out_root.name)
    return out_root / VERIFIER_SPILLS_FILENAME


def _append_ledger(ledger_path: Path, record: Dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def install_exec_spill(sandbox_env_cls: type, *, threshold_bytes: int = SPILL_THRESHOLD_BYTES,
                       ledger_path: Union[str, Path]) -> bool:
    """Wrap ``sandbox_env_cls.exec`` (an ``async def exec(self, cmd, ...)``) with
    the spill (module docstring). Idempotent: a class whose ``exec`` already
    carries :data:`INSTALLED_MARKER` — installed on it or on a parent — is left
    as is and ``False`` is returned; ``True`` means installed now. The wrapper
    keeps the original's signature (``functools.wraps`` → ``__wrapped__``), which
    Inspect's ``SandboxEnvironmentProxy`` inspects to decide whether to pass
    ``concurrency``."""
    original = sandbox_env_cls.exec
    if getattr(original, INSTALLED_MARKER, False):
        return False
    ledger = Path(ledger_path)
    threshold = int(threshold_bytes)

    @functools.wraps(original)
    async def exec_with_spill(self, cmd, *args, **kwargs):
        if not (is_python_c(cmd) and argv_bytes(cmd) > threshold):
            return await original(self, cmd, *args, **kwargs)       # the original argv object, untouched
        program = cmd[2]
        path = spill_path(program)
        n = argv_bytes(cmd)
        # the EXEC channel with the program on stdin (F-18: the provider's write_file is
        # Modal's dead legacy filesystem API), then the ORIGINAL exec on the file with
        # every keyword argument as given. A failed write is a refusal by name, never a
        # run of a missing / truncated file.
        written = await original(self, ["sh", "-c", f"cat > {path}"], input=program, timeout=SPILL_WRITE_TIMEOUT_S)
        if getattr(written, "success", True) is False:
            raise RuntimeError(f"{LOG_PREFIX} could not write the spilled program to {path} over the exec channel "
                               f"(returncode {getattr(written, 'returncode', '?')}; stderr {str(getattr(written, 'stderr', ''))[-300:]!r})")
        _append_ledger(ledger, {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                                "argv_bytes": n, "program_sha256": program_sha256(program), "path": path})
        logger.warning("%s spilled %d bytes -> %s", LOG_PREFIX, n, path)
        return await original(self, [cmd[0], path, *cmd[3:]], *args, **kwargs)

    setattr(exec_with_spill, INSTALLED_MARKER, True)
    sandbox_env_cls.exec = exec_with_spill
    return True


def read_spills(ledger_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Every ledger line as a dict; ``[]`` when the file is absent."""
    p = Path(ledger_path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").split("\n") if line.strip()]


def n_spilled(ledger_path: Union[str, Path]) -> int:
    return len(read_spills(ledger_path))


def rec_field(out_root: Union[str, Path], *, threshold_bytes: int = SPILL_THRESHOLD_BYTES) -> Dict[str, Any]:
    """The launcher REC's ``verifier_transport`` block for the configuration at
    ``out_root``: the threshold this tree spills at, the ledger's count (0
    when no spill was recorded) and the ledger's path (:func:`ledger_path_for_root`:
    the row dir's on a row engine, the store's on a solo run) — the ROW-REC names
    where a spill landed."""
    ledger = ledger_path_for_root(out_root)
    return {"spill_threshold_bytes": int(threshold_bytes), "n_spilled": n_spilled(ledger), "ledger": str(ledger)}
