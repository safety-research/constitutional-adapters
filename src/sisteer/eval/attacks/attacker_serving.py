"""Serving identity for the unsteered vLLM jailbreak attacker.

This module is the shared seam for the shell launcher, Modal launchers, and run
manifests.  It is deliberately stdlib-only so it can run in the dedicated vLLM
environment without installing the rest of sisteer.

Only the attacker may consume these options.  The steered target has a
per-request cache salt and must not compose prefix-caching flags from here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

QUANTIZATION_CHOICES: tuple[str, ...] = ("none", "fp8")
MAMBA_CACHE_MODE = "align"
KV_CACHE_DTYPE = "auto"  # model dtype (bf16 here); do not add an FP8-KV axis
ATTACKER_ENABLE_THINKING = False
MANIFEST_SCHEMA = "attacker_serving/1"
DEFAULTS_REL = Path("configs") / "attacker_serving.json"
_REPO = Path(__file__).resolve().parents[4]
DEFAULTS_PATH = _REPO / DEFAULTS_REL


class AttackerServingError(ValueError):
    """A serving option or identity is missing, malformed, or contradictory."""


@dataclass(frozen=True)
class AttackerServingOptions:
    quantization: str
    prefix_caching: bool

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATION_CHOICES:
            raise AttackerServingError(
                f"attacker_quantization must be one of {QUANTIZATION_CHOICES}, "
                f"got {self.quantization!r}"
            )
        if not isinstance(self.prefix_caching, bool):
            raise AttackerServingError("attacker prefix_caching must be boolean")

    def vllm_argv(self, *, model_role: str = "attacker") -> list[str]:
        """Return explicit vLLM flags, refusing use by a target launcher."""
        if model_role != "attacker":
            raise AttackerServingError(
                "attacker serving flags may only be composed for model_role='attacker'; "
                "the steered target must remain outside this path"
            )
        argv: list[str] = []
        if self.quantization == "fp8":
            argv += ["--quantization", "fp8"]
        if self.prefix_caching:
            argv += ["--enable-prefix-caching", "--mamba-cache-mode", MAMBA_CACHE_MODE]
        else:
            argv += ["--no-enable-prefix-caching"]
        return argv

    def as_dict(self) -> dict:
        out = asdict(self)
        out.update(
            attacker_quantization=self.quantization,
            kv_cache_dtype=KV_CACHE_DTYPE,
            mamba_cache_mode=MAMBA_CACHE_MODE if self.prefix_caching else "none",
            enable_thinking=ATTACKER_ENABLE_THINKING,
        )
        return out


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"on", "true", "1", "yes"}:
            return True
        if norm in {"off", "false", "0", "no"}:
            return False
    raise AttackerServingError(f"{name} must be on/off (true/false), got {value!r}")


def load_defaults(path: str | Path | None = None) -> AttackerServingOptions:
    p = Path(path) if path is not None else DEFAULTS_PATH
    try:
        raw = json.loads(p.read_text())
    except FileNotFoundError as exc:
        raise AttackerServingError(f"attacker serving defaults file missing: {p}") from exc
    except json.JSONDecodeError as exc:
        raise AttackerServingError(f"attacker serving defaults file is not JSON: {p}: {exc}") from exc
    if not isinstance(raw, dict) or set(("quantization", "prefix_caching")) - raw.keys():
        raise AttackerServingError(
            f"{p} must contain quantization and prefix_caching"
        )
    return AttackerServingOptions(
        quantization=str(raw["quantization"]),
        prefix_caching=_as_bool(raw["prefix_caching"], "prefix_caching"),
    )


def resolve(
    quantization: str | None = None,
    prefix_caching=None,
    *,
    defaults_path: str | Path | None = None,
) -> tuple[AttackerServingOptions, dict[str, str]]:
    defaults = load_defaults(defaults_path)
    sources = {"quantization": "defaults_file", "prefix_caching": "defaults_file"}
    q = defaults.quantization
    if quantization not in (None, ""):
        q = str(quantization)
        sources["quantization"] = "flag"
    pc = defaults.prefix_caching
    if prefix_caching not in (None, ""):
        pc = _as_bool(prefix_caching, "prefix_caching")
        sources["prefix_caching"] = "flag"
    return AttackerServingOptions(q, pc), sources


_ENGINE_LINE = re.compile(
    r"Initializing a V1 LLM engine \(v(?P<version>[^)]+)\) with config: (?P<cfg>.*)"
)
_FIELD = {
    "quantization": re.compile(r"(?<![\w.])quantization=(?P<v>[^,\s]+)"),
    "enable_prefix_caching": re.compile(r"\benable_prefix_caching=(?P<v>True|False)"),
    "kv_cache_dtype": re.compile(r"\bkv_cache_dtype=(?P<v>[^,\s]+)"),
    "dtype": re.compile(r"(?<![\w.])dtype=(?P<v>torch\.\w+)"),
    # vllm 0.25.1 prints the window as max_seq_len=; older/newer lines say
    # max_model_len= -- accept either so the manifest records the window.
    "max_model_len": re.compile(r"\bmax_(?:model|seq)_len=(?P<v>\d+)"),
}
_MAMBA_MODE = re.compile(
    r"Mamba cache mode is set to '(?P<v>\w+)'|Prefix caching in Mamba cache '(?P<w>\w+)'"
)
_HIT_RATE = re.compile(r"Prefix cache hit rate: (?P<v>[\d.]+)%")
_KV_TOKENS = re.compile(r"GPU KV cache size: (?P<v>[\d,]+) tokens")


def parse_engine_log(text: str) -> dict:
    out = {
        "engine_config_seen": False,
        "vllm_version": None,
        "quantization": None,
        "enable_prefix_caching": None,
        "kv_cache_dtype": None,
        "dtype": None,
        "max_model_len": None,
        "mamba_cache_mode": None,
        "kv_cache_tokens": None,
        "prefix_cache_hit_rate_last_pct": None,
    }
    matches = list(_ENGINE_LINE.finditer(text))
    if matches:
        match = matches[-1]
        out["engine_config_seen"] = True
        out["vllm_version"] = match.group("version")
        cfg = match.group("cfg")
        for key, regex in _FIELD.items():
            found = regex.search(cfg)
            if found:
                out[key] = found.group("v").strip("'\"")
        if out["quantization"] in {"None", "null"}:
            out["quantization"] = "none"
        if out["enable_prefix_caching"] is not None:
            out["enable_prefix_caching"] = out["enable_prefix_caching"] == "True"
        if out["max_model_len"] is not None:
            out["max_model_len"] = int(out["max_model_len"])
    modes = list(_MAMBA_MODE.finditer(text))
    if modes:
        out["mamba_cache_mode"] = modes[-1].group("v") or modes[-1].group("w")
    hits = list(_HIT_RATE.finditer(text))
    if hits:
        out["prefix_cache_hit_rate_last_pct"] = float(hits[-1].group("v"))
    kv = list(_KV_TOKENS.finditer(text))
    if kv:
        out["kv_cache_tokens"] = int(kv[-1].group("v").replace(",", ""))
    return out


def verify_engine(opts: AttackerServingOptions, engine: dict) -> list[str]:
    problems: list[str] = []
    if not engine.get("engine_config_seen"):
        return ["engine config line not found in serve log"]
    actual_q = engine.get("quantization") or "none"
    if opts.quantization == "none" and actual_q != "none":
        problems.append(f"requested bf16 weights but engine quantization={actual_q}")
    if opts.quantization == "fp8" and not str(actual_q).startswith("fp8"):
        problems.append(f"requested fp8 weights but engine quantization={actual_q}")
    actual_pc = engine.get("enable_prefix_caching")
    if actual_pc is None:
        problems.append("engine did not report enable_prefix_caching")
    elif actual_pc != opts.prefix_caching:
        problems.append(
            f"requested prefix_caching={opts.prefix_caching} but engine reports {actual_pc}"
        )
    actual_kv = engine.get("kv_cache_dtype")
    if actual_kv != KV_CACHE_DTYPE:
        problems.append(
            f"attacker KV cache must report {KV_CACHE_DTYPE}; engine reports {actual_kv}"
        )
    if opts.prefix_caching and engine.get("mamba_cache_mode") != MAMBA_CACHE_MODE:
        problems.append(
            f"requested mamba_cache_mode={MAMBA_CACHE_MODE}; engine reports "
            f"{engine.get('mamba_cache_mode')}"
        )
    return problems


def write_manifest(
    path: str | Path,
    *,
    opts: AttackerServingOptions,
    sources: dict,
    engine: dict,
    model: str,
    served_name: str,
    endpoint: str | None,
    log_path: str | None,
    argv: list[str],
    extra: dict | None = None,
) -> dict:
    problems = verify_engine(opts, engine)
    if problems:
        raise AttackerServingError(
            "attacker server identity could not be verified: " + "; ".join(problems)
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model": model,
        "served_model_name": served_name,
        "endpoint": endpoint,
        "requested": opts.as_dict(),
        "requested_sources": dict(sources),
        "vllm_argv_flags": list(argv),
        "engine": dict(engine),
        "engine_verified": True,
        "log": log_path,
    }
    if extra:
        manifest["extra"] = dict(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(p)
    return manifest


def read_manifest(path: str | Path) -> dict:
    p = Path(path)
    try:
        manifest = json.loads(p.read_text())
    except FileNotFoundError as exc:
        raise AttackerServingError(f"attacker serving manifest missing: {p}") from exc
    except json.JSONDecodeError as exc:
        raise AttackerServingError(f"attacker serving manifest is not JSON: {p}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise AttackerServingError(f"unexpected attacker serving schema in {p}")
    requested = manifest.get("requested") or {}
    quantization = requested.get("attacker_quantization", requested.get("quantization"))
    if quantization not in QUANTIZATION_CHOICES:
        raise AttackerServingError(f"bad attacker quantization in {p}")
    if ("attacker_quantization" in requested
            and requested.get("quantization") != quantization):
        raise AttackerServingError(f"contradictory attacker quantization fields in {p}")
    if not isinstance(requested.get("prefix_caching"), bool):
        raise AttackerServingError(f"bad attacker prefix_caching in {p}")
    expected_mamba = MAMBA_CACHE_MODE if requested["prefix_caching"] else "none"
    if requested.get("mamba_cache_mode") != expected_mamba:
        raise AttackerServingError(f"bad attacker mamba_cache_mode in {p}")
    if requested.get("kv_cache_dtype") != KV_CACHE_DTYPE:
        raise AttackerServingError(f"attacker KV cache identity is not {KV_CACHE_DTYPE} in {p}")
    if requested.get("enable_thinking") is not ATTACKER_ENABLE_THINKING:
        raise AttackerServingError(f"bad attacker thinking-mode identity in {p}")
    if manifest.get("engine_verified") is not True:
        raise AttackerServingError(f"unverified attacker serving manifest: {p}")
    options = AttackerServingOptions(str(quantization), requested["prefix_caching"])
    problems = verify_engine(options, manifest.get("engine") or {})
    if problems:
        raise AttackerServingError(
            f"attacker serving manifest no longer verifies in {p}: " + "; ".join(problems)
        )
    return manifest


SERVING_SOURCE_MANIFEST = "manifest"          # engine-verified at write, re-verified at read
SERVING_SOURCE_POOL = "attacker_pool"         # advertised by the shared pool's own standup


def driver_fields(manifest: dict | None, *, manifest_path: str | None = None) -> dict:
    """Fields copied into ``attacker_engine.json`` by the run driver.

    ``attacker_serving_source`` names where the identity came from: ``manifest``
    (the launcher's engine-verified sidecar, re-verified here), ``attacker_pool``
    (see :func:`pool_driver_fields`), or ``None`` when nothing was recorded --
    then ``attacker_quantization`` is ``'unrecorded'``, loudly, never a guess."""
    if manifest is None:
        return {
            "attacker_quantization": "unrecorded",
            "attacker_prefix_caching": None,
            "attacker_kv_cache_dtype": None,
            "attacker_mamba_cache_mode": None,
            "attacker_enable_thinking": None,
            "attacker_vllm_version": None,
            "attacker_serving_manifest": manifest_path,
            "attacker_serving_source": None,
        }
    requested = manifest["requested"]
    engine = manifest["engine"]
    quant = requested.get("attacker_quantization", requested["quantization"])
    return {
        "attacker_quantization": quant,
        "attacker_prefix_caching": bool(requested["prefix_caching"]),
        "attacker_kv_cache_dtype": engine.get("kv_cache_dtype") or requested["kv_cache_dtype"],
        "attacker_mamba_cache_mode": engine.get("mamba_cache_mode") or requested["mamba_cache_mode"],
        "attacker_enable_thinking": bool(requested["enable_thinking"]),
        "attacker_vllm_version": engine.get("vllm_version"),
        "attacker_serving_manifest": manifest_path,
        "attacker_serving_source": SERVING_SOURCE_MANIFEST,
    }


SERVING_RECORD_KEY = "serving"   # the endpoint record's engine-attested sub-dict


def _engine_quantization(engine: dict) -> str:
    q = engine.get("quantization") or "none"
    return "fp8" if str(q).startswith("fp8") else str(q)


def attested_serving_record(manifest: dict, *, manifest_path: str | None = None) -> dict:
    """The ``serving`` sub-dict a shared pool ADVERTISES, derived from a VERIFIED
    manifest (``write_manifest`` / ``read_manifest`` -- the engine's own
    resolved-config line checked against the requested options).  Every value
    here comes from the ENGINE line, never from a literal in the launcher: a
    pool started with different flags advertises what its engine actually
    resolved, or (on a contradiction) does not start.  ``requested`` and
    ``engine`` ride along so a consumer can re-verify before copying."""
    engine = dict(manifest["engine"])
    requested = dict(manifest["requested"])
    options = AttackerServingOptions(
        str(requested.get("attacker_quantization", requested.get("quantization"))),
        bool(requested["prefix_caching"]))
    problems = verify_engine(options, engine)
    if problems:
        raise AttackerServingError(
            "cannot attest a pool's serving identity: " + "; ".join(problems))
    return {
        "attested": True,
        "attacker_quantization": _engine_quantization(engine),
        "prefix_caching": bool(engine["enable_prefix_caching"]),
        "mamba_cache_mode": engine.get("mamba_cache_mode"),
        "kv_cache_dtype": engine.get("kv_cache_dtype"),
        "vllm_version": engine.get("vllm_version"),
        "requested": requested,
        "engine": engine,
        "manifest": manifest_path,
    }


def pool_driver_fields(serving: dict | None, *, declared_quantization=None,
                       url: str | None = None) -> dict:
    """The driver's ``attacker_engine.json`` fields for a run whose attacker is a
    SHARED pool (``sisteer.eval.attacks.attacker_pool``).

    Only an ENGINE-ATTESTED ``serving`` record (:func:`attested_serving_record`,
    re-verified here: the recorded engine line must still agree with the
    recorded request, and the advertised precision/cache values must be the
    ones that line implies) is copied.  A record that merely DECLARES a
    precision -- a top-level ``attacker_quantization`` literal with no attested
    block, the shape the first pool advertised -- is REFUSED, not copied: a
    declaration can contradict the engine and would then sit beside genuinely
    verified shards as if it were one.  A pool that advertises NOTHING about
    its serving is refused too: a shared pool's standups attest (Modal and
    Slurm alike), so a record without the block is stale, stripped or
    malformed, and its actual precision/cache mode is unverifiable -- the
    ``'unrecorded'`` outcome exists only for the per-job vllm-without-manifest
    path (``driver_fields(None)``), never for a pool."""
    if serving is None:
        if declared_quantization is not None:
            raise AttackerServingError(
                f"shared attacker pool DECLARES attacker_quantization={declared_quantization!r} "
                "but carries no engine-attested serving record; refusing to copy a declaration "
                "(redeploy the pool with the attesting standup)")
        raise AttackerServingError(
            "shared attacker pool advertises no engine-attested serving record at all; its "
            "weight precision and prefix-cache mode are unverifiable -- refusing (a pool's "
            "standup writes the attested block; a record without it is stale, stripped or "
            "malformed)")
    if not isinstance(serving, dict) or serving.get("attested") is not True:
        raise AttackerServingError(
            "shared attacker pool's serving record is not engine-attested; refusing to copy it")
    try:
        engine = dict(serving["engine"])
        requested = dict(serving["requested"])
        options = AttackerServingOptions(
            str(requested.get("attacker_quantization", requested.get("quantization"))),
            bool(requested["prefix_caching"]))
    except (KeyError, TypeError, AttackerServingError) as exc:
        raise AttackerServingError(
            f"shared attacker pool's serving record is incomplete ({type(exc).__name__}: {exc}); "
            "refusing to copy it") from exc
    problems = verify_engine(options, engine)
    if problems:
        raise AttackerServingError(
            "shared attacker pool's serving record no longer verifies: " + "; ".join(problems))
    quant = _engine_quantization(engine)
    prefix = bool(engine["enable_prefix_caching"])
    if serving.get("attacker_quantization") != quant or serving.get("prefix_caching") is not prefix:
        raise AttackerServingError(
            "shared attacker pool's advertised precision/cache values contradict its own engine "
            f"line ({serving.get('attacker_quantization')!r}/{serving.get('prefix_caching')!r} vs "
            f"{quant!r}/{prefix!r}); refusing to copy them")
    if declared_quantization is not None and declared_quantization != quant:
        raise AttackerServingError(
            f"shared attacker pool's top-level attacker_quantization={declared_quantization!r} "
            f"contradicts its attested engine line ({quant!r}); refusing")
    return {
        "attacker_quantization": quant,
        "attacker_prefix_caching": prefix,
        "attacker_kv_cache_dtype": engine.get("kv_cache_dtype"),
        "attacker_mamba_cache_mode": engine.get("mamba_cache_mode"),
        # the driver's OWN request parameter, identical for a pool and a per-job server
        "attacker_enable_thinking": ATTACKER_ENABLE_THINKING,
        "attacker_vllm_version": engine.get("vllm_version"),
        "attacker_serving_manifest": (serving.get("manifest")
                                      or (f"attacker_pool:{url}" if url else "attacker_pool")),
        "attacker_serving_source": SERVING_SOURCE_POOL,
    }


def quantization_identity(manifest: dict) -> str:
    requested = manifest.get("requested") or manifest
    value = requested.get("attacker_quantization", requested.get("quantization"))
    if value not in QUANTIZATION_CHOICES:
        raise AttackerServingError(f"missing/invalid attacker_quantization: {value!r}")
    return str(value)


def assert_uniform_quantization(manifests: Iterable[dict]) -> str | None:
    """Refuse a run/aggregate that mixes bf16 and FP8 attacker text."""
    values = {quantization_identity(manifest) for manifest in manifests}
    if len(values) > 1:
        raise AttackerServingError(
            "refusing to mix attacker quantization modes within one run: "
            + ", ".join(sorted(values))
        )
    return next(iter(values), None)


def http_status(url: str, *, timeout: float = 5.0) -> int:
    """The HTTP status of ``GET url``, or 0 when nothing answered (connection
    refused, timed out, unresolvable host, malformed URL) -- stdlib only.

    The launcher (scripts/vllm_attacker_serve.sh) polls ``/health`` through
    this, under the interpreter that already composes its flags, so readiness
    depends on nothing the serving venv does not provide. It used to shell out
    to ``curl``: a container image without curl (Modal's debian_slim and
    nvidia/cuda bases ship none) read ``000`` forever against a HEALTHY server
    and killed it at the deadline (2026-09-06).
    """
    import http.client
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:          # answered, non-2xx
        return int(exc.code)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return 0


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_options(p):
        p.add_argument("--quantization", choices=QUANTIZATION_CHOICES)
        p.add_argument("--prefix-caching", choices=("on", "off"))
        p.add_argument("--defaults")

    p_argv = sub.add_parser("argv")
    add_options(p_argv)
    p_manifest = sub.add_parser("manifest")
    add_options(p_manifest)
    p_manifest.add_argument("--out", required=True)
    p_manifest.add_argument("--log", required=True)
    p_manifest.add_argument("--model", required=True)
    p_manifest.add_argument("--served-name", required=True)
    p_manifest.add_argument("--endpoint")
    p_manifest.add_argument("--extra-json")
    p_show = sub.add_parser("show")
    p_show.add_argument("--manifest", required=True)
    p_health = sub.add_parser("health")
    p_health.add_argument("--endpoint", required=True)
    p_health.add_argument("--timeout", type=float, default=5.0)
    ns = parser.parse_args(argv)

    try:
        if ns.command == "health":
            # curl's three-digit form (000 = nothing answered); exit 0 either
            # way -- the launcher compares the text, not the rc
            print(f"{http_status(ns.endpoint, timeout=ns.timeout):03d}")
            return 0
        if ns.command == "show":
            print(json.dumps(driver_fields(read_manifest(ns.manifest), manifest_path=ns.manifest), indent=2))
            return 0
        opts, sources = resolve(ns.quantization, ns.prefix_caching, defaults_path=ns.defaults)
        flags = opts.vllm_argv()
        if ns.command == "argv":
            print("\n".join(flags))
            return 0
        engine = parse_engine_log(Path(ns.log).read_text(errors="replace"))
        extra = json.loads(ns.extra_json) if ns.extra_json else None
        manifest = write_manifest(
            ns.out,
            opts=opts,
            sources=sources,
            engine=engine,
            model=ns.model,
            served_name=ns.served_name,
            endpoint=ns.endpoint,
            log_path=ns.log,
            argv=flags,
            extra=extra,
        )
        print(json.dumps(driver_fields(manifest, manifest_path=ns.out)))
        return 0
    except (AttackerServingError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"attacker_serving: REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_cli())
