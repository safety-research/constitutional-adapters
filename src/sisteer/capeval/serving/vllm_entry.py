"""The protocol sampling CLAMP OVERRIDE wrapper entrypoint (bcb-bfcl-200-widen).

A LEAF module — stdlib + ``vllm`` only, no ``sisteer`` import — run in the
SERVING interpreter as ``python -m sisteer.capeval.serving.vllm_entry
--sisteer-max-temp 0.001 serve <model> …`` in place of ``vllm serve <model> …``
(``vllm_server.apply_sampling_clamp_override`` composes it, and ONLY under the
tinycap launcher's ``--sampling protocol``).

WHY. vLLM 0.25.1 ``vllm/sampling_params.py`` pins ``_MAX_TEMP = 1e-2`` and
``SamplingParams.__post_init__`` raises every ``0 < temperature < _MAX_TEMP`` to
0.01 with a nan/inf warning — so a BFCL request for its reference temperature
0.001 (the benchmark's own generation-script default) would EXECUTE at 0.01,
farther from BFCL's near-greedy intent than greedy itself.
The clamp is a module GLOBAL read at call time, so setting it in the server
interpreter before the stock CLI runs lets 0.001 through literally. Numerics:
vLLM divides logits by T and takes a max-subtracted softmax, so at 0.001 every
non-argmax probability underflows to 0 and the argmax gets ~1.0 — stable, equal
to greedy except on ties within ~1e-3 logits; the warning concerns unbounded
logits, which the max-subtraction prevents.

PROCESS REACH. ``SamplingParams`` is a ``msgspec.Struct``, and msgspec re-runs
``__post_init__`` on DECODE, so the clamp would fire again in the engine-core
process that decodes the frontend's ``EngineCoreRequest``. vLLM creates that
process with ``get_mp_context().Process`` — ``fork`` unless
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` or CUDA is initialised in the parent;
the API-server frontend never initialises CUDA — so the forked core (and its
forked workers) inherit the patched module. This wrapper therefore ASSERTS
the fork method (refusing an env that says ``spawn``) and exports it
explicitly. Two more doors the pinned wheel has that an env assertion alone
does not close (code review): ``--api-server-count N > 1`` runs the
frontends through ``APIServerProcessManager`` on ``multiprocessing.get_context
("spawn")`` regardless of the env, and ``--numa-bind`` makes
``_maybe_force_spawn()`` flip the env to ``spawn`` AFTER this assertion; a Ray
executor / data-parallel backend (``--distributed-executor-backend ray`` /
``external_launcher``, ``--data-parallel-backend ray``) runs workers that are
not forked from this interpreter at all. Every one of those topologies is
REFUSED here (:func:`spawn_topology_reason` — the lane never uses them; refusal,
not support), and the effective start method is asserted AT PROCESS CREATION:
:func:`guard_mp_context` wraps the wheel's ``get_mp_context`` so a context whose
start method is not ``fork`` — or an environment that has since been flipped —
refuses before the child exists. Fallback 1, if a live check ever shows a
child re-clamping: a ``vllm.general_plugins`` shim (``load_general_plugins``
runs in every vLLM process). Fallback 2: BFCL is BLOCKED by name in the roster —
never silently greedy, never 0.01.

WHAT IS ASSERTED. The wheel's ``_MAX_TEMP`` must be EXACTLY the pinned 0.01
(a different wheel is a different instrument; refused), the requested value
must lie strictly inside ``(0, 0.01]`` and at or below the protocol value, and
the flag must lead the argv. One line is printed —
``[sisteer] vllm _MAX_TEMP 0.01 -> 0.001 (BFCL protocol)`` — so the server
log states the override, then ``sys.argv`` is rebuilt as ``["vllm", *rest]`` and
the STOCK ``vllm.entrypoints.cli.main.main()`` runs unchanged.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

#: the wheel's clamp (vLLM 0.25.1 ``vllm/sampling_params.py`` ``_MAX_TEMP = 1e-2``)
PINNED_MAX_TEMP = 0.01
#: the value this override is pinned to (``contracts.BFCL_PROTOCOL_TEMPERATURE``;
#: kept as a literal here on purpose — this module imports no sisteer)
PROTOCOL_MAX_TEMP = 0.001
FLAG_MAX_TEMP = "--sisteer-max-temp"
MULTIPROC_ENV = "VLLM_WORKER_MULTIPROC_METHOD"
FORK = "fork"
#: The ``vllm serve`` flags that put a process on a path the fork assertion does
#: not cover (vLLM 0.25.1; code review) — refused, never supported:
#: ``--api-server-count`` above 1 (``vllm/v1/utils.py`` ``APIServerProcessManager``
#: uses ``get_context("spawn")`` unconditionally), ``--numa-bind``
#: (``vllm/utils/system_utils.py`` ``_maybe_force_spawn`` overrides the env to
#: ``spawn``), and the Ray / external-launcher executors and data-parallel
#: backend, whose workers are launched by Ray, not forked from this interpreter.
SPAWN_FORCING_FLAGS: Tuple[str, ...] = ("--numa-bind",)
API_SERVER_COUNT_FLAG = "--api-server-count"
EXECUTOR_BACKEND_FLAG = "--distributed-executor-backend"
DATA_PARALLEL_BACKEND_FLAG = "--data-parallel-backend"
UNFORKED_BACKENDS: Tuple[str, ...] = ("ray", "external_launcher")
#: the SHIM (Slice A's fallback 1): ``vllm_shim/sitecustomize.py`` beside this module —
#: delivered by ``vllm_server.apply_sampling_clamp_override`` through ONE route (``SHIM_ROUTE_ENV``) and ARMED by
#: ``SISTEER_VLLM_MAX_TEMP=<T>``, it re-applies the override in EVERY interpreter vLLM spawns (vLLM
#: forces ``spawn`` for the Qwen3.5 server: its API-server process initialises CUDA before the
#: engine fork). With it armed and importable, a spawn context is PERMITTED; without it, refused.
SHIM_ENV = "SISTEER_VLLM_MAX_TEMP"
SHIM_DIRNAME = "vllm_shim"
SHIM_FILENAME = "sitecustomize.py"
#: the shim's identity line (``vllm_shim/sitecustomize.py`` ``SHIM_SIGNATURE``) — a copy is recognised
#: by CONTENT, never by path (round 8, live: the server imported `sisteer` from the /root mount while
#: PYTHONPATH carried the STAGED tree's shim dir; the staged copy loaded as sitecustomize, and a
#: path-equality check refused it as "another sitecustomize")
SHIM_SIGNATURE = "sisteer-vllm-clamp-override-shim/1"
#: round 8: the SECOND delivery route, robust to a pre-existing sitecustomize earlier on sys.path — a
#: ``.pth`` file the standup drops into the SERVING interpreter's site-packages whose one import line
#: boots the shim by absolute path when the arming variable is set (`site` executes it in every
#: interpreter that uses that site-packages, spawned children included, before any sitecustomize)
SHIM_PTH_NAME = "sisteer_vllm_clamp_shim.pth"
SHIM_PTH_ENV = "SISTEER_VLLM_SHIM_PTH"
#: round 9: the composer installs exactly ONE delivery route and names it here — the
#: `.pth` boot (preferred) or, only when the `.pth` could not be written, the shim dir on PYTHONPATH
#: as `sitecustomize`. Both at once exec the shim twice in one interpreter (two module copies).
SHIM_ROUTE_ENV = "SISTEER_VLLM_SHIM_ROUTE"
#: the marker every clamp finder carries on the instance (the shim's ``MARKER_ATTR``; its value is ``SHIM_SIGNATURE``)
SHIM_MARKER_ATTR = "__sisteer_clamp_shim__"
ROUTE_PTH = "pth"
ROUTE_PYTHONPATH = "pythonpath-sitecustomize"
SHIM_FROM_ATTR = "_SISTEER_MAX_TEMP_FROM"
SHIM_APPLIED_ATTR = "_SISTEER_MAX_TEMP_SHIM"
MECHANISM_IN_PROCESS = "in-process (vllm_entry.apply_override; fork children inherit the patched module)"
MECHANISM_SHIM = ("in-process (vllm_entry.apply_override) + sitecustomize shim armed by SISTEER_VLLM_MAX_TEMP in every "
                  "spawned interpreter")
#: where the pinned wheel defines ``get_mp_context`` (0.25.1: ``vllm.utils.system_utils``;
#: earlier wheels re-exported it from ``vllm.utils``) — both are patched when present, and a
#: wheel exposing it in neither refuses (the guard must reach the process creation)
MP_CONTEXT_MODULES: Tuple[str, ...] = ("vllm.utils.system_utils", "vllm.utils")
OVERRIDE_LINE = "[sisteer] vllm _MAX_TEMP {pinned:g} -> {value:g} (BFCL protocol, D13)"


class ClampOverrideRefused(SystemExit):
    """The override cannot be applied as pinned; the server is NOT started."""

    def __init__(self, why: str) -> None:
        super().__init__(f"[sisteer] vllm_entry: refusing to start the server — {why}")


def parse_argv(argv: Sequence[str]) -> Tuple[float, List[str]]:
    """``(value, rest)``: the flag MUST lead (``--sisteer-max-temp <T> serve …``)
    — the rest is the stock ``vllm`` argv tail, untouched."""
    argv = list(argv)
    if len(argv) < 2 or argv[0] != FLAG_MAX_TEMP:
        raise ClampOverrideRefused(
            f"argv must start with {FLAG_MAX_TEMP} <T>, got {argv[:2]!r}")
    try:
        value = float(argv[1])
    except ValueError:
        raise ClampOverrideRefused(f"{FLAG_MAX_TEMP} {argv[1]!r} is not a float") from None
    if not (0.0 < value <= PINNED_MAX_TEMP):
        raise ClampOverrideRefused(
            f"{FLAG_MAX_TEMP} {value!r} must lie in (0, {PINNED_MAX_TEMP}] — the wheel's clamp is what is lifted")
    if value > PROTOCOL_MAX_TEMP:
        raise ClampOverrideRefused(
            f"{FLAG_MAX_TEMP} {value!r} is above the protocol value {PROTOCOL_MAX_TEMP} (D13): a BFCL request "
            "at 0.001 would still be clamped")
    rest = argv[2:]
    if not rest or rest[0] != "serve":
        raise ClampOverrideRefused(f"the stock argv tail must start with `serve`, got {rest[:1]!r}")
    why = spawn_topology_reason(rest)
    if why is not None:
        raise ClampOverrideRefused(why)
    return value, rest


def _flag_values(argv: Sequence[str], flag: str) -> List[str]:
    """Every value of ``flag`` in ``argv`` (``--flag v`` and ``--flag=v``); a bare
    trailing flag contributes ``""``."""
    out: List[str] = []
    toks = list(argv)
    for i, tok in enumerate(toks):
        if tok == flag:
            out.append(toks[i + 1] if i + 1 < len(toks) else "")
        elif tok.startswith(flag + "="):
            out.append(tok[len(flag) + 1:])
    return out


def spawn_topology_reason(argv: Sequence[str]) -> Optional[str]:
    """Why the ``vllm serve`` argv tail ``argv`` puts a vLLM process on a path the
    override cannot reach — None when every process would be forked from the
    patched interpreter. Refusal, not support (code review): the
    override is certified for the single-frontend forked topology only."""
    toks = list(argv)
    for flag in SPAWN_FORCING_FLAGS:
        if flag in toks or any(t.startswith(flag + "=") for t in toks):
            return (f"{flag} forces the spawn start method after the fork assertion (vLLM "
                    "_maybe_force_spawn): a spawned engine core re-clamps 0.001 to 0.01 on msgspec decode; "
                    "the override is refused under it")
    for value in _flag_values(toks, API_SERVER_COUNT_FLAG):
        try:
            n = int(value)
        except ValueError:
            return f"{API_SERVER_COUNT_FLAG} {value!r} is not an int"
        if n != 1:
            return (f"{API_SERVER_COUNT_FLAG} {n}: the multi-frontend API-server process manager uses "
                    "multiprocessing.get_context('spawn') regardless of the environment, so a second frontend "
                    "would decode requests against the wheel's clamp; the override is refused above 1")
    for flag in (EXECUTOR_BACKEND_FLAG, DATA_PARALLEL_BACKEND_FLAG):
        for value in _flag_values(toks, flag):
            if value in UNFORKED_BACKENDS:
                return (f"{flag} {value}: its workers are launched by {value}, not forked from this interpreter, "
                        "so they construct SamplingParams against the wheel's clamp; the override is refused under it")
    return None


def shim_dir() -> "Path":
    from pathlib import Path
    return Path(__file__).resolve().parent / SHIM_DIRNAME


def shim_armed(environ: Mapping[str, str], value: Optional[float] = None) -> bool:
    """``SISTEER_VLLM_MAX_TEMP`` is set (to ``value`` when given)."""
    raw = environ.get(SHIM_ENV)
    if not raw:
        return False
    try:
        have = float(raw)
    except ValueError:
        return False
    return value is None or have == float(value)


def is_shim_file(path) -> bool:
    """``path`` is a copy of the shim — recognised by its signature line, whatever tree it lives in."""
    try:
        return SHIM_SIGNATURE in open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False


def pth_line(shim_file) -> str:
    """The ONE import line of the ``.pth`` boot: ``site`` executes lines starting with ``import``
    at startup; this one exec's the shim by absolute path in a fresh globals dict (its classes must
    see each other as module globals) iff the arming variable is set. Inert otherwise."""
    p = repr(str(shim_file))
    return (f"import os, sys; exec(compile(open({p}).read(), {p}, 'exec'), {{'__name__': 'sisteer_vllm_clamp_shim', "
            f"'__file__': {p}}}) if os.environ.get({SHIM_ENV!r}) else None")


def pth_target(pth_path) -> Optional[str]:
    """The shim file a ``.pth`` boot line names (None when the file is not ours)."""
    import re as _re
    try:
        text = open(pth_path, encoding="utf-8").read()
    except OSError:
        return None
    m = _re.search(r"compile\(open\('([^']+)'\)", text)
    return m.group(1) if (m and SHIM_ENV in text) else None


def shim_diagnostics(environ: Mapping[str, str]) -> Dict[str, object]:
    """Everything the next live run needs to name the cause (round 8): the arming variable, the
    PYTHONPATH entries that hold a sitecustomize (ours?), what ``find_spec("sitecustomize")``
    resolves to (ours?), what THIS process imported as sitecustomize (ours?), the ``.pth`` route
    (named, present, in this interpreter's site-packages, its target present and ours?), and the
    head of ``sys.path``."""
    import importlib.util
    import os as _os
    import sysconfig
    d: Dict[str, object] = {"armed_env": environ.get(SHIM_ENV), "shim_dir_of_this_module": str(shim_dir())}
    entries = [e for e in (environ.get("PYTHONPATH") or "").split(_os.pathsep) if e]
    d["pythonpath_sitecustomize"] = [{"entry": e, "ours": is_shim_file(_os.path.join(e, SHIM_FILENAME))}
                                     for e in entries if _os.path.isfile(_os.path.join(e, SHIM_FILENAME))]
    try:
        spec = importlib.util.find_spec("sitecustomize")
    except (ImportError, ValueError):
        spec = None
    origin = getattr(spec, "origin", None) if spec is not None else None
    d["find_spec_sitecustomize"] = {"origin": origin, "ours": bool(origin and is_shim_file(origin))}
    loaded = sys.modules.get("sitecustomize")
    lfile = getattr(loaded, "__file__", None) if loaded is not None else None
    d["loaded_sitecustomize"] = {"file": lfile, "ours": bool(lfile and is_shim_file(lfile))} if loaded is not None else None
    pth = environ.get(SHIM_PTH_ENV)
    purelib = sysconfig.get_paths().get("purelib")
    target = pth_target(pth) if pth else None
    d["pth"] = {"path": pth, "present": bool(pth and _os.path.isfile(pth)),
                "in_this_interpreters_site_packages": bool(pth and purelib and _os.path.realpath(_os.path.dirname(pth)) == _os.path.realpath(purelib)),
                "purelib": purelib, "target": target, "target_ours": bool(target and is_shim_file(target))}
    # round 9: the route the composer installed and how many clamp finders THIS interpreter carries —
    # two (a `.pth` boot AND a sitecustomize import of two module copies) was the recursion
    d["route"] = environ.get(SHIM_ROUTE_ENV) or _shim_source(d)[1]
    d["clamp_finders_on_meta_path"] = sum(1 for f in sys.meta_path if getattr(f, SHIM_MARKER_ATTR, None) == SHIM_SIGNATURE)
    d["sys_path_head"] = list(sys.path[:6])
    d["executable"] = sys.executable
    return d


def shim_importable(environ: Mapping[str, str]) -> Optional[str]:
    """Where a spawned child of THIS interpreter would get the shim from — None when it would not.
    Two routes, either suffices: (a) the ``.pth`` boot the standup dropped into this interpreter's
    site-packages (``SISTEER_VLLM_SHIM_PTH`` names it; present, in this interpreter's purelib, its
    target a copy of the shim) — `site` runs it in every child before any sitecustomize; (b) the
    ``sitecustomize`` a child would import — ``find_spec("sitecustomize")`` on this process's path
    (the child inherits the environment and this process's sys.path) resolving to a file that IS
    the shim by signature (ANY copy: mounted or staged), and, when this process already imported a
    sitecustomize, that one being the shim too (else a decoy shadows route (b))."""
    return _shim_source(shim_diagnostics(environ))[0]


def _shim_source(d: Mapping[str, object]) -> Tuple[Optional[str], Optional[str]]:
    """``(where, route)`` a spawned child would get the shim from, read off a :func:`shim_diagnostics`
    dict (the one acceptance rule, shared by ``shim_importable``, ``shim_route`` and the diagnostics'
    own ``route`` field — which is why this reads the dict rather than calling the diagnostics)."""
    pth = d["pth"]
    if pth["present"] and pth["in_this_interpreters_site_packages"] and pth["target_ours"]:
        return str(pth["path"]), ROUTE_PTH
    fs = d["find_spec_sitecustomize"]
    loaded = d["loaded_sitecustomize"]
    if fs["ours"] and (loaded is None or loaded["ours"]):
        return str(fs["origin"]), ROUTE_PYTHONPATH
    return None, None


def spawn_permitted(environ: Mapping[str, str], value: Optional[float] = None) -> bool:
    """A spawned child keeps the override iff the shim is armed AND importable by that child."""
    return shim_armed(environ, value) and shim_importable(environ) is not None


def assert_fork(environ: MutableMapping[str, str], value: Optional[float] = None) -> str:
    """The start method the override can survive. Without the shim: ``VLLM_WORKER_MULTIPROC_METHOD``
    must be absent or ``fork`` and is EXPORTED as ``fork`` (a spawned engine core would re-run
    ``SamplingParams.__post_init__`` against the WHEEL's global and re-clamp 0.001 to 0.01). With
    the shim armed and importable: a spawned child re-applies the override itself, so ``spawn``
    is permitted and the variable is left as vLLM wants it. Returns the method in force
    (``fork``, or the environment's / vLLM's choice under the shim)."""
    have = environ.get(MULTIPROC_ENV)
    if spawn_permitted(environ, value):
        return have or "vllm-default (shim armed: fork or spawn)"
    if have is not None and have != FORK:
        raise ClampOverrideRefused(
            f"{MULTIPROC_ENV}={have!r}: the engine core would be {have}ed and re-clamp the temperature on "
            f"msgspec decode; the override holds only under {FORK} unless the shim is armed and importable; "
            f"diagnostics: {_diag_text(environ)}")
    environ[MULTIPROC_ENV] = FORK
    return FORK


def _diag_text(environ: Mapping[str, str]) -> str:
    import json as _json
    return _json.dumps(shim_diagnostics(environ), default=str, sort_keys=True)


def guard_mp_context(modules: Mapping[str, object], environ: Mapping[str, str]) -> List[str]:
    """Assert the effective start method AT PROCESS CREATION: wrap
    ``get_mp_context`` in every module of ``modules`` that defines it so the
    context handed to vLLM's ``Process(...)`` calls must report ``fork``
    and the environment must still say ``fork`` (``_maybe_force_spawn`` flips it
    behind the assertion in :func:`assert_fork`) — else the server refuses before
    the child exists. Returns the module names patched; refuses when none
    defines it (a wheel whose process-creation seam moved is a different
    instrument: the guard MUST reach it)."""
    patched: List[str] = []
    already = 0
    for name, module in modules.items():
        original = getattr(module, "get_mp_context", None)
        if original is None:
            continue
        if getattr(original, "__sisteer_fork_guard__", False):
            already += 1                                   # guarded earlier in this interpreter: the seam IS reached
            continue

        def guarded(*args, _original=original, **kwargs):
            have = environ.get(MULTIPROC_ENV)
            permitted = spawn_permitted(environ)              # round 6: a spawned child re-applies the override itself
            if have != FORK and not permitted:
                raise ClampOverrideRefused(
                    f"{MULTIPROC_ENV}={have!r} at process creation (flipped after the fork assertion — "
                    "_maybe_force_spawn / a spawn-forcing flag): a spawned child would re-clamp 0.001 to 0.01 "
                    f"(the shim is not armed and importable); diagnostics: {_diag_text(environ)}")
            ctx = _original(*args, **kwargs)
            method = ctx.get_start_method() if hasattr(ctx, "get_start_method") else None
            if method != FORK and not permitted:
                raise ClampOverrideRefused(
                    f"vLLM asked for a {method!r} multiprocessing context at process creation; the override "
                    f"holds only under {FORK} unless the shim is armed and importable; diagnostics: {_diag_text(environ)}")
            return ctx

        guarded.__sisteer_fork_guard__ = True          # idempotent: a second call never double-wraps
        setattr(module, "get_mp_context", guarded)
        patched.append(name)
    if not patched and not already:
        raise ClampOverrideRefused(
            f"none of {list(modules)} defines get_mp_context — the fork guard cannot reach vLLM's process "
            "creation on this wheel (0.25.1 defines it in vllm.utils.system_utils); re-pin before overriding")
    return patched


def _mp_context_modules() -> Dict[str, object]:
    """The importable members of :data:`MP_CONTEXT_MODULES` (each imported once)."""
    import importlib
    out: Dict[str, object] = {}
    for name in MP_CONTEXT_MODULES:
        try:
            out[name] = importlib.import_module(name)
        except ImportError:
            continue
    return out


def apply_override(module, value: float, *, pinned: float = PINNED_MAX_TEMP) -> Mapping[str, object]:
    """Set ``module._MAX_TEMP`` (``vllm.sampling_params``) to ``value`` after
    asserting it IS the pinned wheel's — returns the record ``{from, to}``. When
    the SHIM already applied this exact override in this interpreter (its
    ``_SISTEER_MAX_TEMP_FROM`` names the wheel's original), the wheel's value is
    read from that marker: the same assertion, one step removed."""
    have = getattr(module, "_MAX_TEMP", None)
    shimmed = getattr(module, SHIM_APPLIED_ATTR, None)
    if shimmed is not None:
        original = getattr(module, SHIM_FROM_ATTR, None)
        if original != pinned or shimmed != float(value) or have != float(value):
            raise ClampOverrideRefused(
                f"the sitecustomize shim applied _MAX_TEMP {original!r} -> {shimmed!r} (now {have!r}); this launch "
                f"asks {pinned} -> {value} — two overrides disagree, or the wheel is not the pinned one")
        return {"from": float(pinned), "to": float(value)}
    if have != pinned:
        raise ClampOverrideRefused(
            f"vllm.sampling_params._MAX_TEMP is {have!r}, not the pinned {pinned} of the 0.25.1 wheel — "
            "a different wheel is a different instrument; re-pin before overriding")
    module._MAX_TEMP = float(value)
    return {"from": float(pinned), "to": float(value)}


def shim_route(environ: Mapping[str, str]) -> Optional[str]:
    """The delivery route the composer installed (``SISTEER_VLLM_SHIM_ROUTE``), else what
    :func:`shim_importable` found (a `.pth` in this interpreter's site-packages, or a sitecustomize)."""
    return environ.get(SHIM_ROUTE_ENV) or _shim_source(shim_diagnostics(environ))[1]


def override_mechanism(environ: Mapping[str, str], value: Optional[float] = None) -> str:
    """What holds the override in vLLM's child processes: the shim (spawn-safe; the route named)
    or fork inheritance."""
    if spawn_permitted(environ, value):
        return f"{MECHANISM_SHIM} [route: {shim_route(environ)}]"
    return MECHANISM_IN_PROCESS


def spawned_clamp_probe(conn, temperature: float) -> None:
    """The body of a spawned CHILD: import
    the wheel's ``vllm.sampling_params`` afresh, construct ``SamplingParams(temperature=T)`` and
    report what the clamp resolved it to, plus the module's ``_MAX_TEMP`` and whether the shim
    patched it — through ``conn`` (a multiprocessing pipe), never a shared module object."""
    try:
        import vllm.sampling_params as sp
        params = sp.SamplingParams(temperature=float(temperature))
        conn.send({"ok": True, "temperature": float(params.temperature), "max_temp": float(sp._MAX_TEMP),
                   "shim_applied": getattr(sp, SHIM_APPLIED_ATTR, None), "shim_from": getattr(sp, SHIM_FROM_ATTR, None),
                   "sitecustomize": getattr(sys.modules.get("sitecustomize"), "__file__", None),
                   "start_method": "spawn"})
    except Exception as e:  # noqa: BLE001 — the parent records the failure, it never guesses
        conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def spawned_child_control(*, armed: bool, value: float = PROTOCOL_MAX_TEMP, temperature: float = PROTOCOL_MAX_TEMP,
                          timeout_s: float = 300.0) -> Dict[str, object]:
    """Run :func:`spawned_clamp_probe` in a REAL ``multiprocessing.get_context("spawn")`` child whose
    environment is armed (``SISTEER_VLLM_MAX_TEMP`` set, the shim directory first on PYTHONPATH) or
    not, and return its report. The negative control (``armed=False``) must read the wheel's clamp
    (0.01); the armed child must read ``value`` (0.001). When the serving interpreter also carries
    the ``.pth`` boot the armed child runs BOTH routes — safe since round 9 (one marked finder, no
    cross-delegation), and exactly the combination the live qwen server died of before it."""
    import multiprocessing
    import os as _os
    saved = dict(_os.environ)
    entries = [e for e in (_os.environ.get("PYTHONPATH") or "").split(_os.pathsep) if e and _os.path.realpath(e) != str(shim_dir())]
    try:
        if armed:
            _os.environ[SHIM_ENV] = f"{float(value):g}"
            _os.environ["PYTHONPATH"] = _os.pathsep.join([str(shim_dir()), *entries])
        else:
            _os.environ.pop(SHIM_ENV, None)
            _os.environ["PYTHONPATH"] = _os.pathsep.join(entries)
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=spawned_clamp_probe, args=(child, float(temperature)))
        proc.start()
        child.close()
        report: Dict[str, object] = {"ok": False, "error": f"no report within {timeout_s} s"}
        if parent.poll(timeout_s):
            report = dict(parent.recv())
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
        report.update({"armed": bool(armed), "exitcode": proc.exitcode})
        return report
    finally:
        _os.environ.clear()
        _os.environ.update(saved)


def main(argv: Optional[Sequence[str]] = None, *, environ: Optional[MutableMapping[str, str]] = None,
         stdout=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    environ = os.environ if environ is None else environ
    out = sys.stdout if stdout is None else stdout
    value, rest = parse_argv(argv)
    method = assert_fork(environ, value)
    import vllm.sampling_params as sp   # the SERVING interpreter's wheel

    rec = apply_override(sp, value)
    guarded = guard_mp_context(_mp_context_modules(), environ)
    print(OVERRIDE_LINE.format(pinned=rec["from"], value=rec["to"]), file=out, flush=True)
    print(f"[sisteer] vllm get_mp_context guarded ({method}) in {guarded}; override mechanism: "
          f"{override_mechanism(environ, value)}", file=out, flush=True)
    print(f"[sisteer] shim diagnostics: {_diag_text(environ)}", file=out, flush=True)
    sys.argv = ["vllm", *rest]
    from vllm.entrypoints.cli.main import main as vllm_main   # the stock CLI, unchanged

    result = vllm_main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
