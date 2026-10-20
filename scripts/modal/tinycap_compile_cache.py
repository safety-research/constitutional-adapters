"""tinycap-shared-standup (d): the warm compile cache — ONE tarball per (model, vLLM pin, row shape) on the volume,
unpacked to node-local disk at container start, with a *hit* attested from vLLM's own log.

Why a tarball per row SHAPE and not an image layer or a file-for-file mirror
----------------------------------------------------------------------------
vLLM 0.25.1 keys its compile cache directory ``$VLLM_CACHE_ROOT/torch_compile_cache/<hash>`` on
``[env_hash, config_hash, code_hash, compiler_hash]`` (``vllm/compilation/backends.py`` 1055–1064); ``config_hash``
is ``VllmConfig.compute_hash()``, which folds in ``lora_config.compute_hash()`` — ``max_lora_rank`` AND ``max_loras``
are factors (``vllm/config/lora.py`` 93–105) — and ``scheduler_config.compute_hash()`` — ``max_num_batched_tokens``
(``vllm/config/scheduler.py`` 215–226). A row that serves 8 adapters therefore compiles a DIFFERENT graph from the
same row serving 7 (the negative-control smoke), so a layer per (model, image) would miss for every shape it was
not warmed at, and a GPU image-build step would need the staged tree, the adapter bank and the weights at build
time. Seedrep's ``COMPILE_CACHES`` file-for-file persist (10k–35k inodes per persist on a volume with ~40k free;
``tests/test_seedrep_f5_compile_cache_env_guard.py``) is the anti-pattern this module replaces: here the volume
holds ``index.json`` plus at most :data:`MAX_TARBALLS` tarballs — never more than :data:`MAX_INODES` inodes at ANY
instant, the transient ``.tmp`` of a write included (≤ 8).

``env_hash`` is a factor too — and it carried the per-launch API token
------------------------------------------------------------------------
The first factor, ``env_hash`` = ``hash_factors(envs.compile_factors())`` (``vllm/envs.py`` 2044–2130), is the
sha256 of the sorted JSON of EVERY known vLLM env var's VALUE minus an ignore list — and at 0.25.1 that list does
NOT contain ``VLLM_API_KEY`` (``envs.py`` 752; upstream ``main`` has since added it: "Credential; never affects
compiled artifacts and must not be persisted in cache_key_factors.json"). The launcher mints a fresh random token
per launch (``vllm_server.build_server_launch``) and the engine/worker child inherits it, so landing 1's warm probe
and row hashed apart (``485efd7135`` vs ``71450296d1``) with every config factor equal — a ``restored_miss``.
The fix lives in the worker extension every lora-native worker imports before its first compile
(``native_worker``: the import pops :data:`VLLM_API_KEY_ENV` from the compiling process; the API server keeps
its copy for both auth guards); this module keeps the MODEL of that factor — :func:`hash_env_factors` (vLLM's
digest, byte for byte) and :func:`env_factors_as_the_worker_compiles` — so a test can state, over two synthetic
``cache_key_factors.json`` env blocks, that the scrub is what makes them hash equal. The four tarballs landing 1
persisted under key version ``v1`` recorded hash directories no post-fix standup can print (and carry the
probe's token inside their ``cache_key_factors.json``); :data:`KEY_VERSION` ``v2`` makes them unreachable, so
they age out by LRU and every shape re-warms once.

The served shape must be the key's, or the composer's recorded adjustment of it
-------------------------------------------------------------------------------
The key is minted BEFORE Popen from the CONFIGURED shape, but the composer may lower S (the LoRA kernel index
ceiling: qwen's ``S 45 -> 31``, gemma's ``36 -> 31``; recorded in ``endpoint.json`` ``geometry.adjustments`` as
``{kind, from_S, to_S, D, ceiling}``), and ``max_num_seqs`` / ``max_num_batched_tokens`` / ``max_model_len`` are
``SchedulerConfig`` / ``CompilationConfig`` / ``ModelConfig`` factors of the graph vLLM actually compiled. That
step is a pure function of the key's own fields (the model's pinned D, L, the configured S —
``expected_geometry_for`` pre-computes it), so a configured-shape key names exactly one served graph; what must
never happen is a tarball filed under a key whose served shape the key does not determine. So
:func:`key_inputs_disagreements` holds the key to the READY endpoint's ``geometry`` block (as it holds the LoRA
fields to ``lora_config``) and AGREES when the served shape is the key's, or the endpoint's recorded adjustment
chain leads from the key's S to it (``from_S`` = the key's S, each step from where the last left off, the key's
budget the composed ``S × L``); a served shape the chain does not explain — or one with no recorded step (a
KV-derived S is not a function of the key) — refuses the persist by name (``key_inputs_disagree``). The index
entry records :func:`served_geometry` beside ``inputs`` so a later ``restored_miss`` is readable against the
shape the tarball's graph was compiled for.

``cache_key_factors.json`` is shipped REDACTED
----------------------------------------------
vLLM writes the raw env factors — the token among them — to ``<local_cache_dir>/cache_key_factors.json``
(``backends.py`` 1116–1127) and never reads the file back: its only reference in the 0.25.1 tree is that write,
guarded by ``if not os.path.exists``; a hit is decided from ``vllm_compile_cache.py`` and the inductor artifacts
(``CompilerManager.initialize_cache`` / ``load``). :func:`compile_cache_persist` therefore ships the file with
every :data:`CREDENTIAL_ENV_FACTORS` value replaced by :data:`REDACTED` (:func:`redact_factors`; an unparseable
copy is not shipped at all — vLLM rewrites it when absent), records what it redacted, and the restore still hits.

``restored`` is not ``hit``
---------------------------
:func:`compile_cache_restore` reports whether a tarball was unpacked (``restored``). Whether vLLM USED it is decided
after readiness by :func:`compile_cache_evidence` from two lines of ``vllm_server.log``:

* ``Using cache directory: <root>/torch_compile_cache/<hash>/rank_<r>_<dp>/<prefix> for vLLM's torch.compile``
  (backends.py 1089–1092) — printed on COLD and WARM starts alike, so it evidences only WHICH directory vLLM chose;
  the ``<hash>`` must equal the one the index recorded for the key when the tarball was persisted;
* ``Directly load the compiled graph(s) for compile range <range> from the cache, took <s> s`` (backends.py 292–297)
  — printed only when a compiled graph was loaded from that directory. A cold start prints
  ``Cache the graph of compile range <range> for later use`` (377–380) and ``Compiling a graph for compile range
  <range> takes <s> s`` (392–397) instead.

``hit`` iff a tarball was restored AND the observed hash equals the recorded one AND at least one graph-load line is
present; otherwise ``hit: false`` with the reason. A cold compile also prints ``Using cache directory``; that line
alone is never evidence.

Persist
-------
:func:`compile_cache_persist` tars the node-local cache dirs into ``<key>.tar.gz`` (written as ``.tmp`` and
``os.replace``d), records the hash directory read off THIS container's log in the index, and evicts least-recently-
used entries so the directory never exceeds the bound. It refuses — recorded, never silent — after a restore that
HIT (a hit never rewrites), before readiness, under ``--no-cache-persist``, for a smoke (``--inspect-limit``) or
negative-control shape, when the log names no cache directory, when vLLM reports its cache disabled, and when the
named directory does not exist under the node-local root (the env hook was not honoured).

A ``restored_miss`` RE-PERSISTS. Without this, a key whose tarball's recorded ``vllm_hash_dir`` can no longer be
printed — the token leaving ``env_hash``, an image / vLLM / env bump — would restore on every row, miss, and never
re-persist: a permanent trap. So when a tarball was restored AND the verdict is ``restored_miss`` /
``hash_mismatch`` (vLLM printed a different ``Using cache directory`` hash than the index recorded for the key)
AND the compile completed (at least one ``Compiling a graph … takes`` line), persist REPLACES the stale tarball
under the same key — the fresh tree in, the stale ``<old hash>/`` subtree the restore left node-local NOT shipped,
the same ``.tmp`` + ``os.replace``, the index entry rewritten with the new hash, ``served_geometry`` and a
``replaced_stale`` block naming the old hash / sha256 / age — still subject to the served-geometry agreement and
the redaction. A ``hit``, a matching hash without a graph-load line, an adopted entry with no recorded hash, and a
mismatch whose compile did not complete keep refusing (``restored``, the verdict named).

This module imports only the standard library so the launcher process can import it as a sibling copy layer
(the ``tinycap_images`` pattern) and the tests load it by path without a ``sisteer`` import.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

__all__ = [
    "DEFAULT_VOL_ROOT", "LOCAL_ROOT_DEFAULT", "INDEX_NAME", "INDEX_VERSION", "KEY_VERSION",
    "MAX_INODES", "MAX_TARBALLS", "CACHE_SUBDIRS", "CACHE_ENV_VARS",
    "CACHE_DIR_RE", "GRAPH_LOAD_RE", "GRAPH_COMPILE_RE", "GRAPH_STORE_RE", "CACHE_DISABLED_RE",
    "VLLM_API_KEY_ENV", "CREDENTIAL_ENV_FACTORS", "FACTORS_FILE", "REDACTED",
    "cache_key", "cache_key_inputs", "cache_key_of", "key_inputs_disagreements", "served_geometry",
    "hash_env_factors", "env_factors_as_the_worker_compiles", "redact_factors",
    "compile_cache_env", "with_compile_cache_env",
    "compile_cache_restore", "compile_cache_persist", "compile_cache_evidence", "persist_refusal",
    "read_index", "index_entry", "inode_count", "parse_vllm_log",
]

# --------------------------------------------------------------------------- #
# layout                                                                       #
# --------------------------------------------------------------------------- #

#: The compile-cache directory on the volume: ``contracts.VOL_CACHE_ROOT + "/compile"`` (``/vol/tinycap/cache`` is
#: the launcher's cache root; spelled here so this module needs no ``sisteer`` import — a test pins the equality).
DEFAULT_VOL_ROOT = "/vol/tinycap/cache/compile"
#: The node-local root the three caches unpack into (rule 3: never file-for-file on the volume).
LOCAL_ROOT_DEFAULT = "/tmp/compile_cache"
INDEX_NAME = "index.json"
INDEX_VERSION = 1
#: Folded into every key: bump when the key's field set changes — or when no existing tarball can answer its key
#: any more — so an old tarball never answers a new key. ``v2`` (2026-09-29, the warm-cache-miss fix): the worker
#: extension now drops ``VLLM_API_KEY`` before the first compile, which changes vLLM's ``env_hash`` — every ``v1``
#: tarball's recorded ``vllm_hash_dir`` is one no post-fix standup can print, and since a restore REFUSES the
#: persist (``restored``) a ``v1`` key could never be re-warmed. Landing 1's four tarballs are therefore
#: unreachable by construction and age out by LRU; every shape re-warms once.
KEY_VERSION = "v2"

#: The inode bound (one tarball per (model, vLLM pin, shape); ≤ 8 on the volume) — held at EVERY
#: instant: ``index.json`` + :data:`MAX_TARBALLS` tarballs = 7 steady, and the ONE transient ``.tmp`` a write needs
#: (a tarball's, or the index's own atomic rewrite) makes 8. The plan's "LRU ≤ 7" would put the transient at 9.
MAX_INODES = 8
MAX_TARBALLS = MAX_INODES - 2

#: Node-local sub-directory per cache and the environment variable that points each library at it.
#: ``VLLM_CACHE_ROOT`` is read off vLLM 0.25.1 (``vllm/envs.py`` 644–648: default ``~/.cache/vllm``);
#: ``TRITON_CACHE_DIR`` and ``FLASHINFER_WORKSPACE_BASE`` are Triton's and FlashInfer's documented overrides for
#: ``~/.cache/triton`` and ``~/.cache/flashinfer`` (the two directories seedrep mirrored file-for-file).
CACHE_SUBDIRS: Mapping[str, str] = {"vllm": "VLLM_CACHE_ROOT", "triton": "TRITON_CACHE_DIR",
                                    "flashinfer": "FLASHINFER_WORKSPACE_BASE"}
CACHE_ENV_VARS: Sequence[str] = tuple(CACHE_SUBDIRS.values())
TARBALL_SUFFIX = ".tar.gz"
TMP_SUFFIX = ".tmp"

# --------------------------------------------------------------------------- #
# the log lines (vLLM 0.25.1, vllm/compilation/backends.py)                    #
# --------------------------------------------------------------------------- #

#: backends.py 1089–1092: ``"Using cache directory: %s for vLLM's torch.compile"`` with ``local_cache_dir`` =
#: ``<VLLM_CACHE_ROOT>/torch_compile_cache/<hash>/rank_<rank>_<dp_rank>/<prefix>`` (1073–1080).
CACHE_DIR_RE = re.compile(r"Using cache directory: (?P<dir>\S+) for vLLM's torch\.compile")
#: The ``<hash>`` segment of that path (``hashlib.sha256(str(factors)).hexdigest()[:10]``, 1063).
CACHE_HASH_RE = re.compile(r"(?P<root>.*?)/torch_compile_cache/(?P<hash>[0-9a-f]+)(?:/|$)")
#: backends.py 292–297: printed once per compile range when the compiled graph(s) were LOADED from the cache.
GRAPH_LOAD_RE = re.compile(r"Directly load the compiled graph\(s\) for compile range (?P<range>\([^)]*\)) "
                           r"from the cache, took (?P<s>[0-9.]+) s")
#: backends.py 392–397: printed once per compile range after a COLD compile.
GRAPH_COMPILE_RE = re.compile(r"Compiling a graph for compile range (?P<range>\([^)]*\)) takes (?P<s>[0-9.]+) s")
#: backends.py 377–380: the first graph of a cold compile was stored for later use.
GRAPH_STORE_RE = re.compile(r"Cache the graph of compile range (?P<range>\([^)]*\)) for later use")
#: backends.py 1086: ``CompilationMode.NONE`` / ``VLLM_DISABLE_COMPILE_CACHE`` — nothing is ever cached.
CACHE_DISABLED_RE = re.compile(r"vLLM's torch\.compile cache is disabled\.")


# --------------------------------------------------------------------------- #
# the env_hash factor (vLLM 0.25.1 envs.compile_factors) and its persisted copy #
# --------------------------------------------------------------------------- #

#: The per-launch API token's variable — ``vllm_auth.TOKEN_ENV`` and ``native_worker.TOKEN_ENV``, spelled here so
#: this module needs no ``sisteer`` import (a test pins the three equal). Its VALUE is a compile factor at 0.25.1
#: (``envs.py`` 752, not in ``compile_factors()``'s ignore list); the worker extension pops it before compiling.
VLLM_API_KEY_ENV = "VLLM_API_KEY"
#: The env factors whose value is a credential — 0.25.1 ``envs.py``'s three ``os.environ.get(<name>, None)``
#: secrets (752, 758–759) — never shipped in a tarball on the shared volume.
CREDENTIAL_ENV_FACTORS: Sequence[str] = ("VLLM_API_KEY", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")
#: ``backends.py`` 1116: ``<local_cache_dir>/cache_key_factors.json`` — ``{"env": <raw compile_factors()>,
#: "config_hash", "code_hash", "compiler_hash"}``, written once when absent, never read back by vLLM.
FACTORS_FILE = "cache_key_factors.json"
#: What a redacted credential value reads as in the shipped copy (the key NAME stays, so the file still says
#: which factors entered the hash).
REDACTED = "<redacted>"


def hash_env_factors(env_factors: Mapping[str, Any]) -> str:
    """vLLM's ``hash_factors`` (``vllm/config/utils.py`` 365–367), byte for byte: the sha256 hex digest of
    ``json.dumps(items, sort_keys=True)`` — what ``env_hash`` is over ``compile_factors()``'s dict (the ``env``
    block of :data:`FACTORS_FILE`)."""
    return hashlib.sha256(json.dumps(dict(env_factors), sort_keys=True).encode()).hexdigest()


def env_factors_as_the_worker_compiles(env_factors: Mapping[str, Any]) -> dict:
    """The factor dict as the compiling process now sees it: the same block with :data:`VLLM_API_KEY_ENV` at
    ``None`` — ``compile_factors()``'s getter is ``os.environ.get("VLLM_API_KEY", None)`` and ``normalize_value
    (None)`` is ``None``, so after ``native_worker``'s import-time pop this is the value every launch hashes."""
    out = dict(env_factors)
    out[VLLM_API_KEY_ENV] = None
    return out


def redact_factors(data: bytes) -> tuple[Optional[bytes], list]:
    """``(shipped bytes | None, redacted names)`` for a :data:`FACTORS_FILE`: every :data:`CREDENTIAL_ENV_FACTORS`
    entry of its ``env`` block that holds a non-empty string is replaced by :data:`REDACTED` (names sorted); a
    ``null`` or empty value is not a secret and stays; a file without an ``env`` block ships as is; a file that is
    not JSON ships as ``None`` — excluded — with ``["unparseable"]`` (vLLM rewrites it when absent)."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, ["unparseable"]
    env = doc.get("env") if isinstance(doc, Mapping) else None
    if not isinstance(env, Mapping):
        return data, []
    names = sorted(n for n in CREDENTIAL_ENV_FACTORS if isinstance(env.get(n), str) and env.get(n))
    if not names:
        return data, []
    env = dict(env)
    for n in names:
        env[n] = REDACTED
    doc = dict(doc)
    doc["env"] = env
    return json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"), names


# --------------------------------------------------------------------------- #
# the key                                                                      #
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(s: Any) -> str:
    return _SLUG_RE.sub("-", str(s)).strip("-") or "x"


def cache_key_inputs(model_key: str, vllm_pin: str, max_loras: int, max_lora_rank: int,
                     max_num_batched_tokens: Optional[int] = None, max_model_len: Optional[int] = None,
                     max_num_seqs: Optional[int] = None, *, image_tag: str = "base",
                     lora_target_modules: Optional[Iterable[str]] = None,
                     extra: Optional[Mapping[str, Any]] = None) -> dict:
    """The canonical field set a key digests — every value that changes the graph vLLM compiles for a row
    (the factors of ``VllmConfig.compute_hash`` we control per row):

    * ``model_key`` — the served checkpoint's key (the tarball is per model);
    * ``vllm_pin`` — ``__version__`` is the first factor of ``VllmConfig.compute_hash`` (``vllm/config/vllm.py``
      401); ``image_tag`` names the image variant (``nemotron`` builds its own kernels: a different compiler/code
      hash);
    * ``max_loras`` (the registered adapter count: the served fitted cells + the zero adapter — known before Popen
      from the served set) and ``max_lora_rank`` (``vllm_server.fitted_max_lora_rank`` over the served adapters):
      ``lora_config.compute_hash`` factors; ``lora_target_modules`` too (``tuple(sorted(target_modules))``);
    * ``max_num_batched_tokens`` — ``scheduler_config.compute_hash``'s factor; the lora-native launch composes it
      as ``max_num_seqs * max_model_len`` (``vllm_server.py`` 62 / 304), which is the default here when the caller
      passes ``None``;
    * ``max_model_len`` (a ``model_config`` factor) and ``max_num_seqs`` (the cudagraph capture list is a pure
      function of S: ``cudagraph_capture_sizes_for``; every capture size is a ``compilation_config`` factor);
    * ``extra`` — any further row-shape fact the caller wants keyed (sorted, JSON-canonical).

    Raises ``ValueError`` when a required shape field is missing or not a positive int, so a key is never minted
    from a half-known shape.
    """
    def _pos_int(name: str, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"cache_key: {name} must be a positive int, got {v!r}")
        return int(v)

    if not model_key or not str(model_key).strip():
        raise ValueError("cache_key: model_key is required")
    if not vllm_pin or not str(vllm_pin).strip():
        raise ValueError("cache_key: vllm_pin is required")
    L = _pos_int("max_model_len", max_model_len)
    S = _pos_int("max_num_seqs", max_num_seqs)
    inputs = {
        "key_version": KEY_VERSION,
        "model_key": str(model_key),
        "vllm_pin": str(vllm_pin),
        "image_tag": str(image_tag or "base"),
        "max_loras": _pos_int("max_loras", max_loras),
        "max_lora_rank": _pos_int("max_lora_rank", max_lora_rank),
        "max_model_len": L,
        "max_num_seqs": S,
        "max_num_batched_tokens": (S * L if max_num_batched_tokens is None
                                   else _pos_int("max_num_batched_tokens", max_num_batched_tokens)),
        "lora_target_modules": (None if lora_target_modules is None
                                else sorted(str(m) for m in lora_target_modules)),
        "extra": (None if not extra else json.loads(json.dumps(dict(extra), sort_keys=True, default=str))),
    }
    return inputs


def cache_key(model_key: str, vllm_pin: str, max_loras: int, max_lora_rank: int,
              max_num_batched_tokens: Optional[int] = None, max_model_len: Optional[int] = None,
              max_num_seqs: Optional[int] = None, *, image_tag: str = "base",
              lora_target_modules: Optional[Iterable[str]] = None,
              extra: Optional[Mapping[str, Any]] = None) -> str:
    """OUR name for a row shape's tarball — ``<model>__vllm<pin>__<image>__<16 hex>``; vLLM's own hash decides the
    hit. A readable prefix so a volume listing says what each tarball is, and a digest over
    :func:`cache_key_inputs` so any shape change (``max_loras``, ``max_lora_rank``, S, L, batched tokens, target
    modules, pin, image) is a different key. Filesystem-safe by construction."""
    inputs = cache_key_inputs(model_key, vllm_pin, max_loras, max_lora_rank, max_num_batched_tokens,
                              max_model_len, max_num_seqs, image_tag=image_tag,
                              lora_target_modules=lora_target_modules, extra=extra)
    return cache_key_of(inputs)


def cache_key_of(inputs: Mapping[str, Any]) -> str:
    """:func:`cache_key` from an already-built :func:`cache_key_inputs` record — the same name, byte for byte — so a
    caller that records the inputs (the row runner's ``compile_cache.inputs``) builds them once."""
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"{_slug(inputs['model_key'])}__vllm{_slug(inputs['vllm_pin'])}__{_slug(inputs['image_tag'])}__{digest}"


#: ``endpoint.json`` ``geometry`` field -> key field: the SERVED scheduler shape (``geometry.py`` ``batched_tokens``
#: is vLLM's ``max_num_batched_tokens``; a record spelling it vLLM's way is read too).
_GEOMETRY_FIELDS: Sequence[tuple[str, tuple[str, ...]]] = (
    ("max_num_seqs", ("max_num_seqs",)),
    ("max_num_batched_tokens", ("batched_tokens", "max_num_batched_tokens")),
    ("max_model_len", ("max_model_len",)),
)


def served_geometry(endpoint: Mapping[str, Any]) -> Optional[dict]:
    """The READY endpoint's served scheduler shape — ``geometry.max_num_seqs`` / ``batched_tokens`` (vLLM's
    ``max_num_batched_tokens``; a record spelling it vLLM's way is read too) / ``max_model_len`` — and the
    composer's recorded ``adjustments`` (each ``{kind, from_S, to_S, ...}``); ``None`` when the endpoint carries no
    ``geometry`` block. Recorded in the index entry at persist beside ``inputs``, so a later ``restored_miss`` can
    be read against the shape the tarball's graph was compiled for."""
    g = endpoint.get("geometry") if isinstance(endpoint, Mapping) else None
    if not isinstance(g, Mapping):
        return None
    out: dict = {}
    for field, names in _GEOMETRY_FIELDS:
        v = next((g[n] for n in names if g.get(n) is not None), None)
        out[field] = None if v is None else int(v)
    out["adjustments"] = [dict(a) for a in (g.get("adjustments") or []) if isinstance(a, Mapping)]
    return out


def _explained_by_adjustments(inputs: Mapping[str, Any], served: Mapping[str, Any]) -> Optional[dict]:
    """The served shape the endpoint's recorded adjustment chain EXPLAINS from the key's configured shape: walk
    ``adjustments`` from the key's ``max_num_seqs`` (each step's ``from_S`` must be where the previous left it),
    with the key's budget the composed ``S × L`` so the served ``to_S × L`` follows. ``None`` when there is no
    step with both ends, the chain does not start at the key's S, or the key's budget is not ``S × L``."""
    steps = [a for a in served.get("adjustments") or [] if a.get("from_S") is not None and a.get("to_S") is not None]
    S, L = inputs.get("max_num_seqs"), inputs.get("max_model_len")
    if not steps or S is None or L is None or int(inputs.get("max_num_batched_tokens", -1)) != int(S) * int(L):
        return None
    cur = int(S)
    for a in steps:
        if int(a["from_S"]) != cur:
            return None
        cur = int(a["to_S"])
    return {"max_num_seqs": cur, "max_num_batched_tokens": cur * int(L), "max_model_len": int(L)}


def key_inputs_disagreements(inputs: Mapping[str, Any], endpoint: Mapping[str, Any]) -> list:
    """The key fields a READY standup's ``endpoint.json`` contradicts: ``lora_config.max_loras`` /
    ``max_lora_rank`` / ``target_modules`` against the pre-Popen inputs, then the SERVED geometry
    (:func:`served_geometry`: ``max_num_seqs`` / ``max_num_batched_tokens`` / ``max_model_len``, the shape the
    composer launched and so the ``SchedulerConfig`` / ``CompilationConfig`` / ``ModelConfig`` factors vLLM
    hashed). The geometry AGREES when each served value is the key's, or the one the endpoint's recorded
    ``adjustments`` chain leads to from the key's S (:func:`_explained_by_adjustments`: the LoRA kernel index
    ceiling is a pure function of the key's fields, so the c9513c2 qwen row — key S 45, served 31 under
    ``lora_kernel_index_ceiling 45 -> 31`` — agrees); anything else is named per field, with the recorded chain
    (or its absence). Empty when they agree (or the endpoint carries neither block). A persist under a
    disagreement is refused (``key_inputs_disagree``): the tarball would be filed under a key that does not
    determine the shape served, and the next row of that key would restore a graph compiled for another."""
    out = []
    if not isinstance(endpoint, Mapping):
        return out
    lc = endpoint.get("lora_config")
    if isinstance(lc, Mapping):
        for field in ("max_loras", "max_lora_rank"):
            if field in lc and lc[field] is not None and int(lc[field]) != int(inputs.get(field, -1)):
                out.append(f"{field}: key {inputs.get(field)!r} != endpoint {lc[field]!r}")
        tm = lc.get("target_modules")
        if tm is not None and inputs.get("lora_target_modules") is not None \
                and sorted(str(m) for m in tm) != list(inputs["lora_target_modules"]):
            out.append(f"lora_target_modules: key {inputs['lora_target_modules']!r} != endpoint {sorted(tm)!r}")
    served = served_geometry(endpoint)
    if served is not None:
        explained = _explained_by_adjustments(inputs, served)
        chain = "; ".join(f"{a.get('kind')}: S {a.get('from_S')} -> {a.get('to_S')}" for a in served["adjustments"])
        for field, _names in _GEOMETRY_FIELDS:
            v = served.get(field)
            if v is None or v == int(inputs.get(field, -1)) or (explained is not None and v == explained[field]):
                continue
            why = ""
            if field != "max_model_len":
                why = (f" (recorded adjustment {chain} does not explain it)" if chain
                       else " (no recorded adjustment from the key's shape)")
            out.append(f"{field}: key {inputs.get(field)!r} != endpoint {v!r}{why}")
    return out


# --------------------------------------------------------------------------- #
# the env hook                                                                 #
# --------------------------------------------------------------------------- #

def compile_cache_env(local_root: str | os.PathLike = LOCAL_ROOT_DEFAULT, *, mkdir: bool = True) -> dict:
    """The three cache variables, each under ``local_root`` (``VLLM_CACHE_ROOT=<root>/vllm``,
    ``TRITON_CACHE_DIR=<root>/triton``, ``FLASHINFER_WORKSPACE_BASE=<root>/flashinfer``); the directories are
    created so a MISS starts on empty node-local dirs, never on ``~/.cache`` (rule 3)."""
    root = Path(local_root)
    out = {}
    for sub, var in CACHE_SUBDIRS.items():
        d = root / sub
        if mkdir:
            d.mkdir(parents=True, exist_ok=True)
        out[var] = str(d)
    return out


def with_compile_cache_env(env: Mapping[str, str], local_root: str | os.PathLike = LOCAL_ROOT_DEFAULT) -> dict:
    """A COPY of ``env`` (the standup subprocess's environment, as ``_run_configuration`` composes it) with the
    three cache variables set — the hook the row runner applies before ``Popen(standup_argv, env=...)``. The
    caller's mapping is never mutated."""
    out = dict(env)
    out.update(compile_cache_env(local_root))
    return out


# --------------------------------------------------------------------------- #
# the index                                                                    #
# --------------------------------------------------------------------------- #

def _index_path(vol_root: str | os.PathLike) -> Path:
    return Path(vol_root) / INDEX_NAME


def _tarball_path(vol_root: str | os.PathLike, key: str) -> Path:
    return Path(vol_root) / f"{key}{TARBALL_SUFFIX}"


def read_index(vol_root: str | os.PathLike) -> dict:
    """``{"version": 1, "entries": {key: {...}}}``; an absent index reads as empty. A CORRUPT index raises
    ``ValueError`` (the callers record ``index_unreadable`` rather than guessing)."""
    p = _index_path(vol_root)
    if not p.exists():
        return {"version": INDEX_VERSION, "entries": {}}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"compile-cache index unreadable at {p}: {type(e).__name__}: {e}") from e
    if not isinstance(data, Mapping) or not isinstance(data.get("entries"), Mapping):
        raise ValueError(f"compile-cache index malformed at {p}: expected {{'version', 'entries'}}")
    return {"version": data.get("version", INDEX_VERSION), "entries": dict(data["entries"])}


def index_entry(vol_root: str | os.PathLike, key: str) -> Optional[dict]:
    try:
        return read_index(vol_root)["entries"].get(key)
    except ValueError:
        return None


def _write_index(vol_root: str | os.PathLike, index: Mapping[str, Any]) -> None:
    """Atomic: ``index.json.tmp`` then ``os.replace`` — the ONE transient inode the bound reserves for it."""
    p = _index_path(vol_root)
    tmp = p.with_name(p.name + TMP_SUFFIX)
    tmp.write_text(json.dumps({"version": INDEX_VERSION, "entries": index["entries"]}, indent=1, sort_keys=True))
    os.replace(tmp, p)


def inode_count(vol_root: str | os.PathLike) -> int:
    """Every directory entry under ``vol_root``, recursively (the layout is flat; anything nested counts too so a
    regression that unpacks onto the volume is visible)."""
    root = Path(vol_root)
    if not root.exists():
        return 0
    n = 0
    for _dirpath, dirnames, filenames in os.walk(root):
        n += len(dirnames) + len(filenames)
    return n


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _last_used(entry: Mapping[str, Any]) -> float:
    return float(entry.get("last_used_at") or entry.get("written_at") or 0.0)


def _reconcile(vol_root: Path, index: MutableMapping[str, Any]) -> dict:
    """Bring the index and the directory into agreement before any write — the volume is shared by concurrent row
    containers and a lost index update must never leave an uncounted inode: a stale ``*.tmp`` is removed; a
    tarball without an entry is ADOPTED (key = its file name, sha256 recomputed, ``vllm_hash_dir`` unknown, so it
    can be restored but never attested a hit until re-persisted); an entry without a tarball is dropped.
    Returns what was done.

    A ``.tmp`` this container can SEE is always a crashed write: a Modal volume shows a peer container's files only
    after that peer's ``VOL.commit()``, and :func:`compile_cache_persist` commits nothing itself — the caller
    commits once, after persist returns — so a peer's in-flight ``.tmp`` is never visible here. On a plain shared
    filesystem (not this module's deployment) the same removal would race a concurrent writer."""
    entries = index["entries"]
    done = {"removed_tmp": [], "adopted": [], "dropped": []}
    if not vol_root.exists():
        return done
    for p in sorted(vol_root.iterdir()):
        if p.name.endswith(TMP_SUFFIX):
            p.unlink(missing_ok=True)
            done["removed_tmp"].append(p.name)
        elif p.name.endswith(TARBALL_SUFFIX):
            key = p.name[: -len(TARBALL_SUFFIX)]
            if key not in entries:
                st = p.stat()
                entries[key] = {"tarball": p.name, "sha256": _sha256_file(p), "bytes": st.st_size,
                                "vllm_hash_dir": None, "inputs": None, "written_at": st.st_mtime,
                                "last_used_at": st.st_mtime, "adopted": True}
                done["adopted"].append(key)
    for key in list(entries):
        if not (vol_root / entries[key].get("tarball", f"{key}{TARBALL_SUFFIX}")).exists():
            del entries[key]
            done["dropped"].append(key)
    return done


def _evict_to(vol_root: Path, index: MutableMapping[str, Any], n_keep: int, *, protect: str = "") -> list:
    """Delete least-recently-used tarballs until at most ``n_keep`` entries remain (``protect`` is never evicted).
    The index is rewritten by the caller."""
    entries = index["entries"]
    evicted = []
    victims = sorted((k for k in entries if k != protect), key=lambda k: (_last_used(entries[k]), k))
    while len(entries) > n_keep and victims:
        k = victims.pop(0)
        (vol_root / entries[k].get("tarball", f"{k}{TARBALL_SUFFIX}")).unlink(missing_ok=True)
        del entries[k]
        evicted.append(k)
    return evicted


# --------------------------------------------------------------------------- #
# restore                                                                      #
# --------------------------------------------------------------------------- #

def _safe_members(tar: tarfile.TarFile, dest: Path) -> list:
    """Every member must land under ``dest`` as a regular file or directory: no absolute names, no ``..``, no
    links, no devices — the tarballs are ours, but a wrong one must not be able to write outside the node-local
    root."""
    out = []
    dest_r = dest.resolve()
    for m in tar.getmembers():
        name = m.name
        if name.startswith("/") or name.startswith("..") or "/../" in name or os.path.isabs(name):
            raise ValueError(f"compile-cache tarball member escapes the root: {name!r}")
        target = (dest / name).resolve()
        if target != dest_r and dest_r not in target.parents:
            raise ValueError(f"compile-cache tarball member escapes the root: {name!r}")
        if not (m.isfile() or m.isdir()):
            raise ValueError(f"compile-cache tarball member is not a file or directory: {name!r} ({m.type!r})")
        out.append(m)
    return out


def compile_cache_restore(key: str, *, vol_root: str | os.PathLike = DEFAULT_VOL_ROOT,
                          local_root: str | os.PathLike = LOCAL_ROOT_DEFAULT) -> dict:
    """Unpack the key's tarball from the volume to ``local_root`` when the index names it and its sha256 matches;
    otherwise leave the (created, empty) node-local dirs in place. ALWAYS returns the env to hand the standup and
    NEVER decides a hit — ``restored`` says a tarball was unpacked, :func:`compile_cache_evidence` says whether
    vLLM used it.

    ``{"restored": bool, "key", "tarball", "sha256", "bytes", "files", "restore_s", "expected_vllm_hash_dir",
    "reason", "local_root", "env"}`` — ``reason`` ∈ ``restored | no_index | index_unreadable | no_tarball |
    tarball_missing | sha256_mismatch | tarball_unsafe | extract_failed``.

    The caller ``VOL.reload()``s before this (a peer container may have persisted since this container started)
    and passes the returned ``env`` (or :func:`with_compile_cache_env`) to the standup's ``Popen``.
    """
    vol, local = Path(vol_root), Path(local_root)
    env = compile_cache_env(local)
    rec: dict = {"restored": False, "key": key, "tarball": None, "sha256": None, "bytes": 0, "files": 0,
                 "restore_s": None, "expected_vllm_hash_dir": None, "reason": None,
                 "local_root": str(local), "env": env}
    if not _index_path(vol).exists():
        rec["reason"] = "no_index"
        return rec
    try:
        entry = read_index(vol)["entries"].get(key)
    except ValueError as e:
        rec.update(reason="index_unreadable", error=str(e))
        return rec
    if not entry:
        rec["reason"] = "no_tarball"
        return rec
    tarball = vol / entry.get("tarball", f"{key}{TARBALL_SUFFIX}")
    rec.update(tarball=str(tarball), expected_vllm_hash_dir=entry.get("vllm_hash_dir"))
    if not tarball.exists():
        rec["reason"] = "tarball_missing"
        return rec
    t0 = time.perf_counter()
    got = _sha256_file(tarball)
    rec.update(sha256=got, bytes=tarball.stat().st_size)
    if entry.get("sha256") and got != entry["sha256"]:
        rec.update(reason="sha256_mismatch", expected_sha256=entry["sha256"])
        return rec
    try:
        with tarfile.open(tarball, "r:gz") as tar:
            members = _safe_members(tar, local)
            kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(local, members=members, **kwargs)
    except ValueError as e:
        rec.update(reason="tarball_unsafe", error=str(e))
        return rec
    except (OSError, tarfile.TarError) as e:
        rec.update(reason="extract_failed", error=f"{type(e).__name__}: {e}")
        return rec
    rec.update(restored=True, reason="restored", files=sum(1 for m in members if m.isfile()),
               restore_s=round(time.perf_counter() - t0, 3))
    # the LRU clock: a restore is a use (best effort — a peer's concurrent rewrite may win; the entry itself is
    # untouched, so a lost touch costs at most an earlier eviction)
    try:
        index = read_index(vol)
        if key in index["entries"]:
            index["entries"][key]["last_used_at"] = time.time()
            _write_index(vol, index)
    except (OSError, ValueError):
        pass
    return rec


# --------------------------------------------------------------------------- #
# evidence                                                                     #
# --------------------------------------------------------------------------- #

def _log_text(vllm_server_log: str | os.PathLike | None) -> str:
    """A ``Path`` (or a path-like) is read; a ``str`` that names an existing file is read; any other ``str`` IS the
    log text (so a test can hand an excerpt and a runner can hand ``srv / "vllm_server.log"``)."""
    if vllm_server_log is None:
        return ""
    if isinstance(vllm_server_log, os.PathLike):
        p = Path(vllm_server_log)
        return p.read_text(errors="replace") if p.exists() else ""
    s = str(vllm_server_log)
    if "\n" not in s and len(s) < 4096 and os.path.exists(s):
        return Path(s).read_text(errors="replace")
    return s


def parse_vllm_log(vllm_server_log: str | os.PathLike | None) -> dict:
    """The cache facts a ``vllm_server.log`` states, nothing decided: ``cache_dir`` (the first ``Using cache
    directory`` path), ``cache_root`` (its ``VLLM_CACHE_ROOT`` part), ``observed_hash`` / ``observed_hashes``,
    ``graphs_loaded`` / ``load_s`` (the ``Directly load`` lines), ``graphs_compiled`` / ``compile_s`` (the
    ``Compiling a graph … takes`` lines), ``graphs_stored`` (``Cache the graph``), ``cache_disabled``."""
    text = _log_text(vllm_server_log)
    dirs = [m.group("dir") for m in CACHE_DIR_RE.finditer(text)]
    hashes, roots = [], []
    for d in dirs:
        hm = CACHE_HASH_RE.search(d)
        if hm:
            if hm.group("hash") not in hashes:
                hashes.append(hm.group("hash"))
            roots.append(hm.group("root"))
    loads = list(GRAPH_LOAD_RE.finditer(text))
    compiles = list(GRAPH_COMPILE_RE.finditer(text))
    return {
        "cache_dir": dirs[0] if dirs else None,
        "cache_root": roots[0] if roots else None,
        "observed_hash": hashes[0] if hashes else None,
        "observed_hashes": hashes,
        "graphs_loaded": len(loads),
        "load_s": round(sum(float(m.group("s")) for m in loads), 3),
        "load_ranges": [m.group("range") for m in loads],
        "graphs_compiled": len(compiles),
        "compile_s": round(sum(float(m.group("s")) for m in compiles), 3),
        "graphs_stored": len(GRAPH_STORE_RE.findall(text)),
        "cache_disabled": bool(CACHE_DISABLED_RE.search(text)),
    }


def compile_cache_evidence(vllm_server_log: str | os.PathLike | None, key: str, *,
                           restored: bool = False, expected_hash: Optional[str] = None,
                           vol_root: Optional[str | os.PathLike] = None,
                           local_root: Optional[str | os.PathLike] = None) -> dict:
    """The verdict, from vLLM's own log after readiness. ``hit`` iff

    (a) ``restored`` — a tarball was unpacked for this key (the restore record's ``restored``);
    (b) the log's ``Using cache directory: …/torch_compile_cache/<hash>/…`` names the hash the index recorded for
        the key (``expected_hash``; looked up in ``vol_root``'s index when not given); and
    (c) at least one ``Directly load the compiled graph(s) … from the cache`` line is present.

    ``{"hit", "class", "reason", "key", "observed_hash", "expected_hash", "cache_dir", "cache_root",
    "root_matches", "graphs_loaded", "load_s", "graphs_compiled", "compile_s", "cache_disabled", "restored"}``;
    ``class`` ∈ ``hit`` (all three) | ``restored_miss`` (a tarball was unpacked, vLLM did not load from it) |
    ``cold`` (nothing was restored); ``reason`` ∈ ``hit | cache_disabled | no_cache_dir_line | not_restored |
    no_expected_hash | hash_mismatch | no_graph_load_evidence``. ``root_matches`` says whether the printed root is
    the node-local ``VLLM_CACHE_ROOT`` (given ``local_root``) — the env hook's own attestation.
    """
    facts = parse_vllm_log(vllm_server_log)
    if expected_hash is None and vol_root is not None:
        entry = index_entry(vol_root, key)
        expected_hash = entry.get("vllm_hash_dir") if entry else None
    root_matches = None
    if local_root is not None and facts["cache_root"] is not None:
        root_matches = Path(facts["cache_root"]).resolve() == (Path(local_root) / "vllm").resolve()
    if facts["cache_disabled"]:
        reason = "cache_disabled"
    elif facts["observed_hash"] is None:
        reason = "no_cache_dir_line"
    elif not restored:
        reason = "not_restored"
    elif not expected_hash:
        reason = "no_expected_hash"
    elif expected_hash not in facts["observed_hashes"]:
        reason = "hash_mismatch"
    elif facts["graphs_loaded"] < 1:
        reason = "no_graph_load_evidence"
    else:
        reason = "hit"
    hit = reason == "hit"
    return {
        "hit": hit,
        "class": "hit" if hit else ("restored_miss" if restored else "cold"),
        "reason": reason,
        "key": key,
        "observed_hash": facts["observed_hash"],
        "observed_hashes": facts["observed_hashes"],
        "expected_hash": expected_hash,
        "cache_dir": facts["cache_dir"],
        "cache_root": facts["cache_root"],
        "root_matches": root_matches,
        "graphs_loaded": facts["graphs_loaded"],
        "load_s": facts["load_s"],
        "graphs_compiled": facts["graphs_compiled"],
        "compile_s": facts["compile_s"],
        "cache_disabled": facts["cache_disabled"],
        "restored": bool(restored),
    }


# --------------------------------------------------------------------------- #
# persist                                                                      #
# --------------------------------------------------------------------------- #

def persist_refusal(*, restored: bool, ready: bool, cache_persist: bool = True,
                    inspect_limit: Optional[int] = None, negative_control: Any = None) -> Optional[str]:
    """The no-persist rule: the FIRST reason a row must not write a tarball, or ``None``.
    ``no_cache_persist`` (``--no-cache-persist``), ``smoke_shape`` (an ``--inspect-limit`` run: its stores can
    never PASS the gate and its served set may be a subset), ``negative_control`` (the flipped-digest smoke serves
    N−1 adapters: a different ``max_loras``, a shape no production row has), ``not_ready`` (readiness was not
    attested), ``restored`` (a tarball was unpacked for this key — a hit never rewrites, and a restored-but-miss
    is a finding to report, not a cache to overwrite — EXCEPT the ``restored_miss`` / ``hash_mismatch`` with a
    completed compile that :func:`compile_cache_persist` turns into a replacement of the stale tarball)."""
    if not cache_persist:
        return "no_cache_persist"
    if inspect_limit is not None and int(inspect_limit) > 0:
        return "smoke_shape"
    if negative_control:
        return "negative_control"
    if not ready:
        return "not_ready"
    if restored:
        return "restored"
    return None


def _tar_local(local: Path, tmp: Path, *, compresslevel: int, exclude: Sequence[str] = ()) -> tuple[int, int, dict]:
    """``local/{vllm,triton,flashinfer}`` (those that exist) into ``tmp`` as ``<sub>/...`` members, minus every
    subtree named in ``exclude`` (paths relative to ``local``: the stale ``vllm/torch_compile_cache/<old hash>`` a
    restored_miss left beside the fresh tree); returns (files, bytes-uncompressed, redacted) — ``redacted`` maps
    each :data:`FACTORS_FILE` member whose copy was changed to the names :func:`redact_factors` replaced (or
    ``["unparseable"]`` for one it left out). The on-disk original is never touched."""
    files = nbytes = 0
    redacted: dict = {}
    excluded = {Path(e) for e in exclude}
    with tarfile.open(tmp, "w:gz", compresslevel=compresslevel) as tar:
        for sub in CACHE_SUBDIRS:
            d = local / sub
            if not d.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(d):
                dirnames.sort()
                rel_dir = Path(dirpath).relative_to(local)
                if rel_dir in excluded:
                    dirnames[:] = []  # the whole subtree stays out
                    continue
                tar.add(dirpath, arcname=str(rel_dir), recursive=False)
                for fn in sorted(filenames):
                    p = Path(dirpath) / fn
                    if p.is_symlink() or not p.is_file():
                        continue  # a link or a socket is never shipped (and never extracted, see _safe_members)
                    arc = str(rel_dir / fn)
                    if fn == FACTORS_FILE:
                        data, names = redact_factors(p.read_bytes())
                        if names:
                            redacted[arc] = names
                        if data is None:
                            continue
                        info = tar.gettarinfo(p, arcname=arc)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
                        files += 1
                        nbytes += len(data)
                        continue
                    tar.add(p, arcname=arc, recursive=False)
                    files += 1
                    nbytes += p.stat().st_size
    return files, nbytes, redacted


def compile_cache_persist(key: str, *, local_root: str | os.PathLike = LOCAL_ROOT_DEFAULT,
                          vol_root: str | os.PathLike = DEFAULT_VOL_ROOT,
                          vllm_server_log: str | os.PathLike | None,
                          restored: bool, ready: bool = True, cache_persist: bool = True,
                          inspect_limit: Optional[int] = None, negative_control: Any = None,
                          inputs: Optional[Mapping[str, Any]] = None,
                          endpoint: Optional[Mapping[str, Any]] = None,
                          bring_up_s: Optional[float] = None, max_tarballs: int = MAX_TARBALLS,
                          compresslevel: int = 6) -> dict:
    """Tar the node-local caches onto the volume as ``<vol_root>/<key>.tar.gz`` (``.tmp`` then ``os.replace``),
    record the hash directory THIS container's log names, and keep the directory within the bound. The caller
    ``VOL.reload()``s before and ``VOL.commit()``s ONCE after this returns — never in between (a committed ``.tmp``
    would look like a crashed write to a peer's :func:`_reconcile`).

    Refuses — ``{"persisted": False, "skipped": <reason>}``, recorded, never silent — per :func:`persist_refusal`,
    and further when the log names no cache directory (``no_cache_dir_line``: nothing to index, so no future run
    could ever be attested a hit), when vLLM reports its cache disabled (``cache_disabled``), when the named
    directory does not exist under ``local_root/vllm`` (``local_cache_dir_missing``: the standup did not run with
    the hook's ``VLLM_CACHE_ROOT``), when the index is unreadable (``index_unreadable``), and when a READY
    endpoint contradicts the key's inputs (``key_inputs_disagree``; :func:`key_inputs_disagreements`).

    A RESTORED tarball is normally a refusal (``restored``: a hit never rewrites) — except a ``restored_miss`` /
    ``hash_mismatch`` with a completed compile (the module docstring's re-persist rule), which REPLACES the stale
    tarball under the same key: the verdict is derived here from the log and the index entry the way
    :func:`compile_cache_evidence` derives it (the caller hands only ``restored``), the stale
    ``vllm/torch_compile_cache/<old hash>`` subtree is left out of the new tarball, and ``replaced_stale`` names
    the old ``vllm_hash_dir`` / ``sha256`` / ``bytes`` / ``written_at`` / ``age_s`` in the record and the index
    entry. Every other refusal above still applies first, and the served-geometry check and the redaction still
    apply to the replacement.

    Bound: ``index.json`` + ≤ ``max_tarballs`` tarballs steady, and never more than one transient ``.tmp`` — so
    the count is ≤ ``max_tarballs + 2`` (= :data:`MAX_INODES` at the default) at every instant. A new key evicts
    least-recently-used entries FIRST, down to ``max_tarballs − 1``, so the write itself stays inside the bound;
    a re-persist of an existing key (a replacement included) replaces its tarball in place.

    ``{"persisted", "key", "tarball", "sha256", "bytes", "files", "uncompressed_bytes", "tar_s", "vllm_hash_dir",
    "vllm_cache_dir", "evicted", "reconciled", "inodes", "skipped", "refusal_detail", "redacted",
    "served_geometry", "replaced_stale"}`` — ``redacted``: per :data:`FACTORS_FILE` member, the credential factors
    replaced in the shipped copy (:func:`redact_factors`; ``{}`` when nothing was); ``served_geometry``:
    :func:`served_geometry` of the endpoint (``None`` without one), also written to the index entry;
    ``replaced_stale``: the block above for a restored_miss replacement, else ``None`` (also in the index entry).
    """
    local, vol = Path(local_root), Path(vol_root)
    rec: dict = {"persisted": False, "key": key, "tarball": None, "sha256": None, "bytes": 0, "files": 0,
                 "uncompressed_bytes": 0, "tar_s": None, "vllm_hash_dir": None, "vllm_cache_dir": None,
                 "evicted": [], "reconciled": None, "inodes": inode_count(vol), "skipped": None,
                 "refusal_detail": None, "redacted": {},
                 "served_geometry": (served_geometry(endpoint) if endpoint is not None else None),
                 "replaced_stale": None}
    why = persist_refusal(restored=restored, ready=ready, cache_persist=cache_persist,
                          inspect_limit=inspect_limit, negative_control=negative_control)
    facts = parse_vllm_log(vllm_server_log)
    stale: Optional[dict] = None
    if why == "restored":
        # THE RE-PERSIST RULE (module docstring): the one refusal that is lifted, and only for a restored_miss
        # whose hash mismatched with a completed compile — the same verdict compile_cache_evidence files, read
        # here from the log and the index entry because the caller hands only `restored`
        try:
            entry = read_index(vol)["entries"].get(key) or {}
        except ValueError as e:
            rec.update(skipped="index_unreadable", refusal_detail=str(e))
            print(f"[compile-cache] persist {key}: skipped (index_unreadable) {e}", flush=True)
            return rec
        verdict = compile_cache_evidence(vllm_server_log, key, restored=True, expected_hash=entry.get("vllm_hash_dir"))
        if verdict["reason"] == "hash_mismatch" and facts["graphs_compiled"] >= 1:
            written = entry.get("written_at")
            stale = {"vllm_hash_dir": entry.get("vllm_hash_dir"), "sha256": entry.get("sha256"),
                     "bytes": entry.get("bytes"), "written_at": written,
                     "age_s": (round(time.time() - float(written), 3) if written else None),
                     "reason": "hash_mismatch", "observed_hash": facts["observed_hash"],
                     "graphs_compiled": facts["graphs_compiled"], "compile_s": facts["compile_s"]}
            why = None
        elif verdict["reason"] == "hit":
            rec["refusal_detail"] = "verdict hit: a hit never rewrites"
        elif verdict["reason"] == "hash_mismatch":
            rec["refusal_detail"] = ("verdict restored_miss/hash_mismatch but the compile did not complete (no "
                                     "'Compiling a graph … takes' line): the fresh tree is not a compiled graph")
        else:
            rec["refusal_detail"] = (f"verdict {verdict['class']}/{verdict['reason']}: a restored tarball is replaced "
                                     "only after a hash_mismatch with a completed compile")
    if why:
        rec["skipped"] = why
        print(f"[compile-cache] persist {key}: skipped ({why}) {rec['refusal_detail'] or ''}", flush=True)
        return rec
    rec.update(vllm_cache_dir=facts["cache_dir"], vllm_hash_dir=facts["observed_hash"])
    if facts["cache_disabled"]:
        rec["skipped"] = "cache_disabled"
    elif facts["observed_hash"] is None:
        rec["skipped"] = "no_cache_dir_line"
    elif not (local / "vllm" / "torch_compile_cache" / facts["observed_hash"]).is_dir():
        rec["skipped"] = "local_cache_dir_missing"
        rec["refusal_detail"] = (f"{local / 'vllm' / 'torch_compile_cache' / facts['observed_hash']} does not exist; "
                                 f"the log's root is {facts['cache_root']!r}")
    elif inputs is not None and endpoint is not None and key_inputs_disagreements(inputs, endpoint):
        rec["skipped"] = "key_inputs_disagree"
        rec["refusal_detail"] = key_inputs_disagreements(inputs, endpoint)
    if rec["skipped"]:
        print(f"[compile-cache] persist {key}: skipped ({rec['skipped']}) {rec['refusal_detail'] or ''}", flush=True)
        return rec

    vol.mkdir(parents=True, exist_ok=True)
    try:
        index = read_index(vol)
    except ValueError as e:
        rec.update(skipped="index_unreadable", refusal_detail=str(e))
        print(f"[compile-cache] persist {key}: skipped (index_unreadable) {e}", flush=True)
        return rec
    rec["reconciled"] = _reconcile(vol, index)
    keep = max(0, int(max_tarballs)) if key in index["entries"] else max(0, int(max_tarballs) - 1)
    rec["evicted"] = _evict_to(vol, index, keep, protect=key)
    if rec["evicted"] or any(rec["reconciled"].values()):
        _write_index(vol, index)  # the directory changed: say so before the long tar

    final = _tarball_path(vol, key)
    tmp = final.with_name(final.name + TMP_SUFFIX)
    exclude = ()
    if stale is not None:
        if stale["vllm_hash_dir"]:
            exclude = (f"vllm/torch_compile_cache/{stale['vllm_hash_dir']}",)  # the restore's stale tree stays out
        print(f"[compile-cache] persist {key}: replacing the stale tarball (recorded hash {stale['vllm_hash_dir']} -> "
              f"observed {stale['observed_hash']}; written {stale['age_s']} s ago)", flush=True)
    t0 = time.perf_counter()
    try:
        files, nbytes, redacted = _tar_local(local, tmp, compresslevel=compresslevel, exclude=exclude)
        os.replace(tmp, final)
    except (OSError, tarfile.TarError) as e:
        tmp.unlink(missing_ok=True)
        rec.update(skipped="tar_failed", refusal_detail=f"{type(e).__name__}: {e}")
        print(f"[compile-cache] persist {key}: skipped (tar_failed) {e}", flush=True)
        return rec
    tar_s = round(time.perf_counter() - t0, 3)
    sha = _sha256_file(final)
    now = time.time()
    index["entries"][key] = {
        "tarball": final.name, "sha256": sha, "bytes": final.stat().st_size, "files": files,
        "uncompressed_bytes": nbytes, "inputs": (dict(inputs) if inputs is not None else None),
        "vllm_hash_dir": facts["observed_hash"], "vllm_cache_dir": facts["cache_dir"],
        "graphs_compiled": facts["graphs_compiled"], "compile_s": facts["compile_s"],
        "bring_up_s_cold": bring_up_s, "written_at": now, "last_used_at": now, "redacted": redacted,
        "served_geometry": rec["served_geometry"], "replaced_stale": stale,
    }
    _write_index(vol, index)
    rec.update(persisted=True, tarball=str(final), sha256=sha, bytes=final.stat().st_size, files=files,
               uncompressed_bytes=nbytes, tar_s=tar_s, inodes=inode_count(vol), redacted=redacted,
               replaced_stale=stale)
    print(f"[compile-cache] persist {key}: {files} files, {rec['bytes']} bytes -> {final.name} "
          f"(hash dir {facts['observed_hash']}, {tar_s} s; {rec['inodes']} inodes on the volume"
          f"{', evicted ' + ','.join(rec['evicted']) if rec['evicted'] else ''})", flush=True)
    return rec
